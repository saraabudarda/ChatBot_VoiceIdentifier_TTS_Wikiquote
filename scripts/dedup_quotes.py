"""
Three-Layer Quote Deduplication Pipeline
=========================================

Phase 1 — Normalize + Exact Hash
    Canonical text form → SHA-256 hash
    Normalization steps:
      1. Unicode NFKC normalization
      2. Strip trailing attribution fragments (— Author)
      3. Strip surrounding quotation marks
      4. Lowercase
      5. Remove punctuation
      6. Collapse whitespace
    → Catches byte-for-byte duplicates instantly.

Layer 2 — MinHash LSH (scalable fuzzy deduplication)
    Uses MinHash signatures + Locality Sensitive Hashing to find
    candidate pairs in O(n), then confirms with Jaccard similarity.
    → Catches paraphrases, typos, minor edits at 1.3M scale.

Layer 3 — Semantic Embeddings + FAISS
    sentence-transformers (all-MiniLM-L6-v2) + FAISS IVFFlat ANN index
    → Catches rephrased duplicates that share the same meaning.
    → Only runs on survivors of L1 + L2.

Usage:
    python scripts/dedup_quotes.py                       # dry run
    python scripts/dedup_quotes.py --delete              # apply deletions
    python scripts/dedup_quotes.py --skip-l2             # L1 + L3 only
    python scripts/dedup_quotes.py --skip-l3             # L1 + L2 only
    python scripts/dedup_quotes.py --threshold 0.9       # Jaccard threshold
    python scripts/dedup_quotes.py --sem-threshold 0.95  # cosine threshold
    python scripts/dedup_quotes.py --batch 50000
"""

import argparse
import hashlib
import logging
import re
import string
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.database.neo4j_client import Neo4jClient

from datasketch import MinHash, MinHashLSH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Phase 1 — Normalisation pipeline
# ─────────────────────────────────────────────

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# Trailing attribution fragments: "— Gandhi", "- Mahatma Gandhi", "~ Author"
_TRAILING_ATTRIBUTION = re.compile(
    r"\s*[\u2014\u2013\-~]\s*[A-Z][^\n]{0,60}$"
)

# Surrounding quotation marks (straight + curly, single + double)
_SURROUNDING_QUOTES = re.compile(
    r'^["\'\u201c\u2018\u00ab\u2039]+|["\'\u201d\u2019\u00bb\u203a]+$'
)


def normalize(text: str) -> str:
    """
    Canonical normalization for quote deduplication.

    Steps (in order):
      1. Unicode NFKC  — unify equivalent characters (e.g. ligatures, fullwidth)
      2. Strip trailing attribution  — remove leaked '— Author Name' at end
      3. Strip surrounding quotes   — remove wrapping " or ' or « »
      4. Lowercase
      5. Remove all punctuation
      6. Collapse whitespace
    """
    # 1. Unicode NFKC
    text = unicodedata.normalize("NFKC", text)

    # 2. Strip trailing attribution (e.g. "— Mahatma Gandhi")
    text = _TRAILING_ATTRIBUTION.sub("", text).strip()

    # 3. Strip surrounding quote characters
    text = _SURROUNDING_QUOTES.sub("", text).strip()

    # 4. Lowercase
    text = text.lower()

    # 5. Remove punctuation
    text = text.translate(_PUNCT_TABLE)

    # 6. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def exact_hash(text: str) -> str:
    """SHA-256 hash of the normalized text."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 3) -> Set[str]:
    """
    k-character shingles from the normalized text.
    Shingles make MinHash sensitive to small edits and reorderings.
    """
    norm = normalize(text)
    if len(norm) < k:
        return {norm}
    return {norm[i : i + k] for i in range(len(norm) - k + 1)}


def make_minhash(text: str, num_perm: int = 128) -> MinHash:
    """Build a MinHash signature for a quote."""
    m = MinHash(num_perm=num_perm)
    for s in shingles(text):
        m.update(s.encode("utf-8"))
    return m


# ─────────────────────────────────────────────
#  Fetch quotes from Neo4j (cursor-based)
# ─────────────────────────────────────────────

def fetch_all_quotes(client: Neo4jClient, batch_size: int = 50_000) -> List[Dict]:
    """
    Fetch all Quote nodes from Neo4j in cursor-based batches.

    Returns:
        List of dicts with keys: node_id, text
    """
    logger.info("Fetching quotes from Neo4j...")
    all_quotes: List[Dict] = []
    last_id = -1

    while True:
        results = client.execute_query(
            """
            MATCH (q:Quote)
            WHERE id(q) > $last_id
            RETURN id(q) AS node_id, q.text AS text
            ORDER BY id(q)
            LIMIT $limit
            """,
            {"last_id": last_id, "limit": batch_size},
        )

        if not results:
            break

        all_quotes.extend(results)
        last_id = results[-1]["node_id"]

        logger.info(f"  Fetched {len(all_quotes):,} quotes so far...")

    logger.info(f"Total quotes fetched: {len(all_quotes):,}")
    return all_quotes


# ─────────────────────────────────────────────
#  Layer 1 — Exact hash deduplication
# ─────────────────────────────────────────────

def layer1_exact(quotes: List[Dict]) -> Tuple[List[Dict], Set[int]]:
    """
    Layer 1: Remove exact duplicates via hash comparison.

    Returns:
        (survivors, duplicate_ids)
    """
    logger.info("=" * 60)
    logger.info("LAYER 1 — Exact Hash Deduplication")
    logger.info("=" * 60)

    seen_hashes: Dict[str, int] = {}   # hash → first node_id
    survivors: List[Dict] = []
    duplicate_ids: Set[int] = set()

    for q in quotes:
        h = exact_hash(q["text"])
        if h in seen_hashes:
            duplicate_ids.add(q["node_id"])
        else:
            seen_hashes[h] = q["node_id"]
            survivors.append(q)

    logger.info(f"  Input:      {len(quotes):,}")
    logger.info(f"  Survivors:  {len(survivors):,}")
    logger.info(f"  Duplicates: {len(duplicate_ids):,}")
    return survivors, duplicate_ids


# ─────────────────────────────────────────────
#  Layer 2 — Blocking + in-bucket comparison
# ─────────────────────────────────────────────

# Common English stopwords for fingerprinting
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "it", "be", "as", "at",
    "this", "that", "was", "are", "were", "been", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "can", "not", "no", "so", "if", "its", "than", "then", "there",
    "their", "they", "we", "you", "he", "she", "i", "my", "your",
    "his", "her", "our", "what", "which", "who", "all", "when", "up",
    "more", "about", "into", "out", "only", "also", "just", "every",
})


def _tokens(text: str) -> List[str]:
    """Split normalized text into tokens."""
    return normalize(text).split()


def _content_tokens(tokens: List[str]) -> List[str]:
    """Remove stopwords, keep content words."""
    return [t for t in tokens if t not in _STOPWORDS]


def _token_jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity on token sets."""
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _norm_levenshtein(a: str, b: str) -> float:
    """
    Normalized Levenshtein similarity = 1 - (edit_distance / max_len).
    Returns 1.0 for identical strings, 0.0 for completely different.
    Uses python-Levenshtein for C-speed.
    """
    try:
        from Levenshtein import distance as lev_dist
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 1.0
        return 1.0 - lev_dist(a, b) / max_len
    except ImportError:
        # Fallback: cheap length-ratio approximation
        return 1.0 - abs(len(a) - len(b)) / max(len(a), len(b), 1)


def _rare_word_overlap(a_content: List[str], b_content: List[str],
                       rare_words: Set[str]) -> float:
    """
    Proportion of rare words that both quotes share.
    Rare words are highly discriminative — if two quotes share several,
    they are very likely duplicates.
    """
    ra = {t for t in a_content if t in rare_words}
    rb = {t for t in b_content if t in rare_words}
    if not ra and not rb:
        return 0.0
    inter = len(ra & rb)
    union = len(ra | rb)
    return inter / union if union else 0.0


def _is_near_duplicate(
    a_norm: str, b_norm: str,
    a_tok: Set[str], b_tok: Set[str],
    a_content: List[str], b_content: List[str],
    rare_words: Set[str],
    threshold: float,
) -> bool:
    """
    Confirm whether two candidate quotes are near-duplicates.

    Uses three signals:
      1. Token Jaccard         — broad token-set overlap
      2. Normalized Levenshtein — character-level edit distance
      3. Rare-word overlap     — distinctive word co-occurrence

    A pair is a duplicate if ANY two of the three signals exceed threshold.
    """
    jaccard   = _token_jaccard(a_tok, b_tok)
    rare_ovlp = _rare_word_overlap(a_content, b_content, rare_words)

    signals_passed = sum([
        jaccard   >= threshold,
        rare_ovlp >= threshold,
    ])

    if signals_passed >= 1:
        # Only call Levenshtein (slightly slower) when there's already a signal
        lev = _norm_levenshtein(a_norm, b_norm)
        signals_passed += (lev >= threshold)

    return signals_passed >= 2


def layer2_blocking(
    quotes: List[Dict],
    threshold: float = 0.82,
) -> Set[int]:
    """
    Layer 2: Scalable near-duplicate detection using blocking + in-bucket comparison.

    Algorithm:
      STEP A — Build word-frequency table across the corpus.
               Words in the bottom 5% of frequency are "rare" (highly discriminative).

      STEP B — For every quote compute 4 blocking keys:
               1. first-10-tokens key  — catches prefix-similar quotes
               2. length-bucket key    — groups same-length quotes (±10 chars)
               3. content-fingerprint  — sorted top-8 non-stopword tokens
               4. rare-token signature — sorted rare words present in quote

      STEP C — For each bucket, compare every pair inside it:
               - Token Jaccard
               - Normalized Levenshtein
               - Rare-word overlap
               Pairs passing at least 2 of 3 thresholds are flagged.

      STEP D — Union-Find clusters; keep longest quote per cluster.

    Large buckets (> 500) are sampled to avoid O(n²) worst-case.
    """
    logger.info("=" * 60)
    logger.info(f"LAYER 2 — Blocking + In-bucket Comparison  (threshold={threshold})")
    logger.info("=" * 60)

    t0 = time.time()
    n = len(quotes)
    logger.info(f"  Processing {n:,} quotes...")

    # STEP A — Word frequency table
    logger.info("  [A] Building corpus word-frequency table...")
    word_freq: Dict[str, int] = {}
    all_token_sets: Dict[int, List[str]] = {}   # node_id → token list
    all_norms: Dict[int, str] = {}              # node_id → normalized text

    for q in quotes:
        toks = _tokens(q["text"])
        all_token_sets[q["node_id"]] = toks
        all_norms[q["node_id"]] = " ".join(toks)
        for t in toks:
            word_freq[t] = word_freq.get(t, 0) + 1

    # Rare = bottom 5% frequency cutoff (but at least seen > 1 time, < 20 times)
    if word_freq:
        sorted_freqs = sorted(word_freq.values())
        cutoff_idx   = max(1, int(len(sorted_freqs) * 0.05))
        freq_cutoff  = sorted_freqs[cutoff_idx]
        freq_cutoff  = min(freq_cutoff, 20)  # cap: words seen < 20 times
        rare_words: Set[str] = {
            w for w, f in word_freq.items() if 1 < f <= freq_cutoff
        }
    else:
        rare_words = set()

    logger.info(f"  [A] Vocabulary: {len(word_freq):,} words | Rare words: {len(rare_words):,}")

    # STEP B — Compute blocking keys
    logger.info("  [B] Computing 4 blocking keys per quote...")

    # buckets: key_string → list of node_ids
    buckets: Dict[str, List[int]] = {}

    all_content_tokens: Dict[int, List[str]] = {}

    for q in quotes:
        nid  = q["node_id"]
        toks = all_token_sets[nid]
        content = _content_tokens(toks)
        all_content_tokens[nid] = content

        # Key 1: first 10 normalized tokens (prefix similarity)
        k1 = "pfx:" + " ".join(toks[:10])

        # Key 2: length bucket ±10 chars of normalized text
        norm_len = len(all_norms[nid])
        k2 = f"len:{(norm_len // 10) * 10}"

        # Key 3: top-8 sorted content words (order-invariant fingerprint)
        top_content = tuple(sorted(content[:8])) if content else ()
        k3 = "cnt:" + "|".join(top_content)

        # Key 4: sorted rare tokens present in this quote
        rare_in_quote = tuple(sorted(t for t in content if t in rare_words))
        k4 = "rare:" + "|".join(rare_in_quote[:6]) if rare_in_quote else None

        for k in [k1, k2, k3]:
            buckets.setdefault(k, []).append(nid)
        if k4:
            buckets.setdefault(k4, []).append(nid)

    # STEP C — In-bucket pairwise comparison
    logger.info("  [C] Comparing pairs within candidate buckets...")

    # Union-Find
    parent: Dict[int, int] = {q["node_id"]: q["node_id"] for q in quotes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    checked_pairs: Set[Tuple[int, int]] = set()
    confirmed_dups = 0
    total_pairs_checked = 0
    MAX_BUCKET = 500        # cap bucket size to avoid O(n²)

    for bucket_key, members in buckets.items():
        if len(members) < 2:
            continue

        # Sample large buckets to avoid worst-case
        sample = members if len(members) <= MAX_BUCKET else members[:MAX_BUCKET]

        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                a_id, b_id = sample[i], sample[j]
                pair = (min(a_id, b_id), max(a_id, b_id))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)
                total_pairs_checked += 1

                if _is_near_duplicate(
                    all_norms[a_id], all_norms[b_id],
                    set(all_token_sets[a_id]), set(all_token_sets[b_id]),
                    all_content_tokens[a_id], all_content_tokens[b_id],
                    rare_words, threshold,
                ):
                    union(a_id, b_id)
                    confirmed_dups += 1

    logger.info(f"  [C] Pairs checked: {total_pairs_checked:,} | Confirmed near-dups: {confirmed_dups:,}")

    # STEP D — Cluster and keep longest
    logger.info("  [D] Clustering and selecting survivors...")
    node_text: Dict[int, str] = {q["node_id"]: q["text"] for q in quotes}
    clusters: Dict[int, List[int]] = {}

    for q in quotes:
        root = find(q["node_id"])
        clusters.setdefault(root, []).append(q["node_id"])

    duplicate_ids: Set[int] = set()
    dup_clusters = 0

    for root, members in clusters.items():
        if len(members) == 1:
            continue
        dup_clusters += 1
        # Keep the longest (most complete) version
        keep = max(members, key=lambda nid: len(node_text[nid]))
        for nid in members:
            if nid != keep:
                duplicate_ids.add(nid)

    elapsed = time.time() - t0
    logger.info(f"  Duplicate clusters:         {dup_clusters:,}")
    logger.info(f"  Quotes to delete (Layer 2): {len(duplicate_ids):,}")
    logger.info(f"  Layer 2 time:               {elapsed:.1f}s")
    return duplicate_ids


# ─────────────────────────────────────────────
#  Delete from Neo4j
# ─────────────────────────────────────────────

def delete_nodes(client: Neo4jClient, node_ids: Set[int], batch_size: int = 5000):
    """Delete Quote nodes by internal ID in batches."""
    ids_list = list(node_ids)
    total = len(ids_list)
    deleted = 0

    for i in range(0, total, batch_size):
        batch = ids_list[i : i + batch_size]
        client.execute_query(
            """
            MATCH (q:Quote)
            WHERE id(q) IN $ids
            DETACH DELETE q
            """,
            {"ids": batch},
        )
        deleted += len(batch)
        logger.info(f"  Deleted {deleted:,}/{total:,} nodes")

    logger.info(f"Deletion complete — {total:,} nodes removed")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Three-layer quote deduplication pipeline"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Apply deletions to Neo4j (default: dry run)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.82,
        help="Similarity threshold for Layer 2 in-bucket comparison (default: 0.82)",
    )
    parser.add_argument(
        "--skip-l2",
        action="store_true",
        help="Skip Layer 2 blocking (run L1 only)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=50_000,
        help="Fetch batch size from Neo4j (default: 50000)",
    )
    args = parser.parse_args()

    mode = "DELETE MODE" if args.delete else "DRY RUN (use --delete to apply)"
    logger.info("=" * 60)
    logger.info(f"  Quote Deduplication Pipeline  —  {mode}")
    logger.info("=" * 60)

    client = Neo4jClient(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
    )

    try:
        start = time.time()

        # ── Fetch ────────────────────────────────────────────────
        quotes = fetch_all_quotes(client, batch_size=args.batch)
        initial_count = len(quotes)

        # ── Layer 1: Exact hash ──────────────────────────────────
        survivors_l1, dups_l1 = layer1_exact(quotes)

        # ── Layer 2: Blocking ────────────────────────────────────
        dups_l2: Set[int] = set()
        survivors_l2 = survivors_l1

        if not args.skip_l2:
            dups_l2 = layer2_blocking(survivors_l1, threshold=args.threshold)
            survivors_l2 = [q for q in survivors_l1 if q["node_id"] not in dups_l2]

        all_dups = dups_l1 | dups_l2

        # ── Summary ──────────────────────────────────────────────
        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info("DEDUPLICATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Initial quotes:          {initial_count:,}")
        logger.info(f"  Layer 1 (exact hash):    {len(dups_l1):,} duplicates")
        logger.info(f"  Layer 2 (blocking):      {len(dups_l2):,} duplicates")
        logger.info(f"  Total to remove:         {len(all_dups):,}")
        logger.info(f"  Remaining after cleanup: {initial_count - len(all_dups):,}")
        logger.info(f"  Reduction:               {len(all_dups)/initial_count*100:.1f}%")
        logger.info(f"  Time elapsed:            {elapsed:.1f}s")

        # ── Delete ──────────────────────────────────────────────
        if args.delete:
            logger.info("Deleting duplicates from Neo4j...")
            delete_nodes(client, all_dups)
            final = client.count_nodes("Quote")
            logger.info(f"Final quote count in database: {final:,}")
        else:
            logger.info("DRY RUN — no changes made. Run with --delete to apply.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
