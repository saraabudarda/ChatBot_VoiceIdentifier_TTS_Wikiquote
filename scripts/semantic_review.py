"""
Phase 3 — Semantic Duplicate Review Tool
==========================================

DESIGN PRINCIPLE: This tool NEVER auto-deletes anything.
Semantic similarity ≠ duplicate for quotes.

    "To be or not to be"
    "The question is whether one should exist"

These are embedding-near but NOT duplicates — one is the original,
one is a paraphrase or interpretation. Auto-deleting would destroy
valuable data.

Instead, this tool builds a REVIEW QUEUE that a human can inspect:
  - Suspicious candidate pairs (same author, high cosine similarity)
  - Semantic clusters (groups of related quotes)
  - A CSV ready for manual moderation

Pipeline:
  1. Fetch quotes from Neo4j (with author info)
  2. Generate sentence embeddings (all-MiniLM-L6-v2, 384-dim)
  3. Build FAISS IVFFlat index for approximate nearest-neighbor search
  4. Find candidate pairs above cosine threshold
  5. Apply same-author filter (cross-author = influence, not duplicate)
  6. Write review_queue.csv and clusters_report.csv

Usage:
    python scripts/semantic_review.py                        # full run
    python scripts/semantic_review.py --limit 50000          # sample
    python scripts/semantic_review.py --threshold 0.94       # stricter
    python scripts/semantic_review.py --cross-author         # include cross-author pairs
    python scripts/semantic_review.py --output-dir ./review  # output location
"""

import argparse
import csv
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.database.neo4j_client import Neo4jClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Lazy imports (heavy — only load when needed)
# ─────────────────────────────────────────────

def _load_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    """Load sentence-transformers model with a clear error message."""
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        logger.info("Model loaded.")
        return model
    except Exception as e:
        logger.error(
            f"Failed to load sentence-transformers: {e}\n"
            "Run: pip install sentence-transformers"
        )
        sys.exit(1)


def _load_faiss():
    """Load faiss with a clear error message."""
    try:
        import faiss
        return faiss
    except Exception as e:
        logger.error(
            f"Failed to load faiss: {e}\n"
            "Run: pip install faiss-cpu"
        )
        sys.exit(1)


# ─────────────────────────────────────────────
#  Fetch quotes with author information
# ─────────────────────────────────────────────

def fetch_quotes_with_authors(
    client: Neo4jClient,
    limit: Optional[int] = None,
    batch_size: int = 50_000,
) -> List[Dict]:
    """
    Fetch Quote nodes together with their author/source name.

    Returns:
        List of dicts: {node_id, text, author}
    """
    logger.info("Fetching quotes with author info from Neo4j...")
    all_quotes: List[Dict] = []
    last_id = -1
    fetch_limit = limit or float("inf")

    while len(all_quotes) < fetch_limit:
        remaining = int(min(batch_size, fetch_limit - len(all_quotes)))

        # Match quotes to their Person/Source/Work via SAID or HAS_QUOTE
        results = client.execute_query(
            """
            MATCH (q:Quote)
            WHERE id(q) > $last_id
            OPTIONAL MATCH (author)-[:SAID|HAS_QUOTE]->(q)
            RETURN id(q)       AS node_id,
                   q.text      AS text,
                   author.name AS author
            ORDER BY id(q)
            LIMIT $limit
            """,
            {"last_id": last_id, "limit": remaining},
        )

        if not results:
            break

        all_quotes.extend(results)
        last_id = results[-1]["node_id"]
        logger.info(f"  Fetched {len(all_quotes):,} quotes...")

    logger.info(f"Total fetched: {len(all_quotes):,}")
    return all_quotes


# ─────────────────────────────────────────────
#  Embedding generation
# ─────────────────────────────────────────────

def generate_embeddings(
    texts: List[str],
    model,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Generate L2-normalized sentence embeddings in batches.

    Returns:
        float32 numpy array of shape (n, embedding_dim)
    """
    logger.info(f"Generating embeddings for {len(texts):,} quotes...")
    t0 = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize → cosine sim = dot product
    )

    elapsed = time.time() - t0
    logger.info(
        f"Embeddings generated: shape={embeddings.shape}, "
        f"time={elapsed:.1f}s  ({len(texts)/elapsed:.0f} quotes/s)"
    )
    return embeddings.astype("float32")


# ─────────────────────────────────────────────
#  FAISS index
# ─────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray, n_probe: int = 64) -> object:
    """
    Build a FAISS IVFFlat index for fast approximate nearest-neighbor search.

    IVFFlat:
      - Divides vectors into nlist clusters (Voronoi cells)
      - At query time, only searches n_probe nearest cells
      - Orders of magnitude faster than brute-force at 1M scale
      - L2-normalized vectors → inner product = cosine similarity

    Args:
        embeddings: float32 array, shape (n, d)
        n_probe:    number of cells to search at query time (higher = more accurate)
    """
    faiss = _load_faiss()
    n, d = embeddings.shape

    # nlist: ~sqrt(n) clusters, capped for very large datasets
    nlist = min(int(n ** 0.5), 4096)
    nlist = max(nlist, 64)

    logger.info(f"Building FAISS IVFFlat index: n={n:,}, d={d}, nlist={nlist}, nprobe={n_probe}")
    t0 = time.time()

    quantizer = faiss.IndexFlatIP(d)           # Inner product (= cosine on L2-normed)
    index     = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = n_probe

    logger.info(f"FAISS index built in {time.time()-t0:.1f}s  ({index.ntotal:,} vectors)")
    return index


# ─────────────────────────────────────────────
#  Candidate pair extraction
# ─────────────────────────────────────────────

def find_candidate_pairs(
    index,
    embeddings: np.ndarray,
    quotes: List[Dict],
    threshold: float = 0.94,
    top_k: int = 10,
    same_author_only: bool = True,
) -> List[Dict]:
    """
    Query FAISS for each quote's nearest neighbors.
    Filter by cosine threshold and optionally by same author.

    Args:
        index:            FAISS index
        embeddings:       L2-normalized float32 array
        quotes:           List of quote dicts (must match index order)
        threshold:        Cosine similarity threshold
        top_k:            Neighbors to retrieve per query
        same_author_only: If True, only flag same-author pairs

    Returns:
        List of candidate pair dicts for the review queue
    """
    logger.info(
        f"Searching for candidate pairs  "
        f"(threshold={threshold}, same_author_only={same_author_only})..."
    )
    t0 = time.time()

    # Search in batches for memory efficiency
    batch_size = 1024
    n = len(quotes)
    seen_pairs: Set[Tuple[int, int]] = set()
    candidates: List[Dict] = []

    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        batch = embeddings[start:end]

        scores, indices = index.search(batch, top_k + 1)  # +1 to skip self

        for i, (score_row, idx_row) in enumerate(zip(scores, indices)):
            a_pos = start + i
            a     = quotes[a_pos]

            for score, j in zip(score_row, idx_row):
                if j < 0 or j == a_pos:
                    continue                        # invalid or self
                if score < threshold:
                    continue                        # below threshold

                b = quotes[j]

                # Same-author filter
                if same_author_only:
                    a_author = (a.get("author") or "").strip().lower()
                    b_author = (b.get("author") or "").strip().lower()
                    if not a_author or a_author != b_author:
                        continue                    # cross-author → skip

                pair_key = (min(a_pos, j), max(a_pos, j))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                candidates.append({
                    "id_a":        a["node_id"],
                    "id_b":        b["node_id"],
                    "author_a":    a.get("author", "Unknown"),
                    "author_b":    b.get("author", "Unknown"),
                    "similarity":  round(float(score), 4),
                    "text_a":      a["text"],
                    "text_b":      b["text"],
                    "len_a":       len(a["text"]),
                    "len_b":       len(b["text"]),
                    "recommendation": _recommend(float(score), a["text"], b["text"]),
                })

        if (start // batch_size) % 10 == 0:
            logger.info(f"  Processed {end:,}/{n:,} queries, {len(candidates):,} pairs so far...")

    # Sort by similarity descending (most suspicious first)
    candidates.sort(key=lambda x: x["similarity"], reverse=True)

    elapsed = time.time() - t0
    logger.info(f"Found {len(candidates):,} candidate pairs in {elapsed:.1f}s")
    return candidates


def _recommend(similarity: float, text_a: str, text_b: str) -> str:
    """
    Suggest a human action based on similarity score and length difference.

    Does NOT make the final call — that is for the human reviewer.
    """
    len_ratio = min(len(text_a), len(text_b)) / max(len(text_a), len(text_b), 1)

    if similarity >= 0.98 and len_ratio >= 0.90:
        return "LIKELY_DUPLICATE — review and delete shorter"
    elif similarity >= 0.95:
        return "SUSPICIOUS — compare carefully"
    else:
        return "POSSIBLE_VARIANT — may keep both"


# ─────────────────────────────────────────────
#  Semantic clustering
# ─────────────────────────────────────────────

def build_clusters(candidates: List[Dict]) -> Dict[int, List[int]]:
    """
    Union-Find clustering of candidate pairs.
    Returns: {cluster_root: [node_ids]}
    """
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for pair in candidates:
        union(pair["id_a"], pair["id_b"])

    clusters: Dict[int, List[int]] = defaultdict(list)
    for node_id in parent:
        clusters[find(node_id)].append(node_id)

    return {k: v for k, v in clusters.items() if len(v) > 1}


# ─────────────────────────────────────────────
#  Output: CSV files
# ─────────────────────────────────────────────

def write_review_queue(candidates: List[Dict], output_path: Path):
    """Write the review queue CSV — sorted by similarity descending."""
    fieldnames = [
        "similarity", "recommendation",
        "id_a", "author_a", "text_a",
        "id_b", "author_b", "text_b",
        "len_a", "len_b",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    logger.info(f"Review queue written → {output_path}  ({len(candidates):,} pairs)")


def write_clusters_report(
    clusters: Dict[int, List[int]],
    quotes: List[Dict],
    output_path: Path,
):
    """Write cluster summary CSV — one row per cluster member."""
    id_to_quote = {q["node_id"]: q for q in quotes}
    rows = []

    for cluster_id, members in sorted(clusters.items(), key=lambda x: -len(x[1])):
        for node_id in members:
            q = id_to_quote.get(node_id, {})
            rows.append({
                "cluster_id":    cluster_id,
                "cluster_size":  len(members),
                "node_id":       node_id,
                "author":        q.get("author", "Unknown"),
                "text_preview":  (q.get("text", ""))[:120],
            })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cluster_id", "cluster_size", "node_id", "author", "text_preview"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Cluster report written → {output_path}  ({len(clusters):,} clusters)")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Semantic duplicate REVIEW tool (no auto-delete)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max quotes to process (default: all). Use e.g. 50000 to test quickly.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.94,
        help="Cosine similarity threshold for flagging pairs (default: 0.94)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Nearest neighbors to retrieve per quote (default: 10)",
    )
    parser.add_argument(
        "--cross-author",
        action="store_true",
        help="Also flag cross-author pairs (default: same-author only)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./review_output",
        help="Directory for output CSV files (default: ./review_output)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=50_000,
        help="Neo4j fetch batch size (default: 50000)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Phase 3 — Semantic Duplicate Review Tool")
    logger.info("  ⚠️  READ-ONLY: No data will be deleted")
    logger.info("=" * 60)
    if args.limit:
        logger.info(f"  Mode: SAMPLE ({args.limit:,} quotes)")
    else:
        logger.info("  Mode: FULL DATABASE")
    logger.info(f"  Cosine threshold:   {args.threshold}")
    logger.info(f"  Same-author only:   {not args.cross_author}")
    logger.info(f"  Output directory:   {output_dir.resolve()}")
    logger.info("=" * 60)

    start = time.time()

    # ── Connect ──────────────────────────────────────────────────
    client = Neo4jClient(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE,
    )

    try:
        # ── 1. Fetch quotes ──────────────────────────────────────
        quotes = fetch_quotes_with_authors(
            client,
            limit=args.limit,
            batch_size=args.batch,
        )

        if not quotes:
            logger.error("No quotes found. Is Neo4j running and populated?")
            return

        texts = [q["text"] for q in quotes]

        # ── 2. Embeddings ────────────────────────────────────────
        model = _load_sentence_transformer(args.model)
        embeddings = generate_embeddings(texts, model)

        # ── 3. FAISS index ───────────────────────────────────────
        index = build_faiss_index(embeddings)

        # ── 4. Candidate pairs ───────────────────────────────────
        candidates = find_candidate_pairs(
            index=index,
            embeddings=embeddings,
            quotes=quotes,
            threshold=args.threshold,
            top_k=args.top_k,
            same_author_only=not args.cross_author,
        )

        # ── 5. Clusters ──────────────────────────────────────────
        clusters = build_clusters(candidates)

        # ── 6. Write output ──────────────────────────────────────
        review_path   = output_dir / "review_queue.csv"
        clusters_path = output_dir / "clusters_report.csv"

        write_review_queue(candidates, review_path)
        write_clusters_report(clusters, quotes, clusters_path)

        # ── Summary ──────────────────────────────────────────────
        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info("PHASE 3 COMPLETE — REVIEW SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Quotes processed:       {len(quotes):,}")
        logger.info(f"  Suspicious pairs found: {len(candidates):,}")
        logger.info(f"  Semantic clusters:      {len(clusters):,}")
        logger.info(f"  Time elapsed:           {elapsed:.1f}s")
        logger.info("")
        logger.info("  Next steps:")
        logger.info(f"    1. Open {review_path}")
        logger.info("    2. Sort by 'similarity' descending")
        logger.info("    3. Review 'LIKELY_DUPLICATE' rows first")
        logger.info("    4. Manually delete confirmed duplicates via Neo4j Browser")
        logger.info("       or add their IDs to a delete list")
        logger.info("")
        logger.info("  ⚠️  No data was modified by this tool.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
