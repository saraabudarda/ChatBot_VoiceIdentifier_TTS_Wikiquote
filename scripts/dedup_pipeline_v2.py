"""
Track 2 — Scalable Deduplication Pipeline
===========================================

Identifies exact and near-duplicate quotes, clusters them, and selects a
canonical "best" quote for each cluster.

This script is strictly NON-DESTRUCTIVE. It does not delete Quote nodes.
Instead, it sets the following properties on Quote nodes:
  - duplicate_group_id
  - duplicate_type      ("exact" or "near")
  - is_canonical        (boolean)
  - canonical_quote_id  (ID of the chosen original)
  - quote_hash

Pipeline Stages:
  Stage 1: Exact Duplicate Detection (Normalized Hashing)
  Stage 2: Near Duplicate Detection (Blocking + Jaccard/Levenshtein)
  Stage 3: Canonical Selection (Ranking by relationships/score)
"""

import argparse
import hashlib
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Fast C-extension for distance
try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    print("Please install python-Levenshtein: pip install python-Levenshtein")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.database.neo4j_client import Neo4jClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Normalization (Stage 1 Prep)
# ─────────────────────────────────────────────

_PUNCT_TABLE = str.maketrans("", "", r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”‘’«»…""")

def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not text:
        return ""
    # Strip wrapping quotes
    text = re.sub(r'^["\'\u201c\u2018]+|["\'\u201d\u2019]+$', '', text.strip()).strip()
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def exact_hash(normalized_text: str) -> str:
    """Deterministic hash for O(1) exact duplicate grouping."""
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()[:16]


# ─────────────────────────────────────────────
#  Near-Duplicate Blocking (Stage 2)
# ─────────────────────────────────────────────

def _tokens(text: str) -> List[str]:
    return text.split()

def _norm_levenshtein(s1: str, s2: str) -> float:
    if not s1 and not s2: return 1.0
    dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (dist / max_len)

def _jaccard(set1: Set[str], set2: Set[str]) -> float:
    if not set1 and not set2: return 1.0
    if not set1 or not set2: return 0.0
    return len(set1 & set2) / len(set1 | set2)


def generate_blocking_keys(normalized_text: str, tokens: List[str]) -> List[str]:
    """
    Generate multiple keys to group potential near-duplicates.
    If two quotes share ANY key, they will be compared.
    """
    if len(tokens) < 3 or len(normalized_text) < 15:
        return [] # Too short to safely fuzzy match

    keys = []
    # Key 1: Exact first 5 tokens (catches prefix similarity)
    keys.append("pfx:" + " ".join(tokens[:5]))

    # Key 2: Exact last 5 tokens (catches suffix similarity)
    keys.append("sfx:" + " ".join(tokens[-5:]))

    # Key 3: Length bucket + sorted 3 longest words (stops alignment mismatch)
    longest_words = sorted(tokens, key=len, reverse=True)[:3]
    length_bucket = (len(normalized_text) // 10) * 10
    keys.append(f"len{length_bucket}:" + "_".join(sorted(longest_words)))

    return keys


# ─────────────────────────────────────────────
#  Canonical Selection (Stage 3)
# ─────────────────────────────────────────────

def select_canonical(cluster: List[Dict]) -> Dict:
    """
    Rank quotes in a cluster and return the best one.
    Ranking criteria (highest weight first):
      +1000 if has_person (SAID)
      +500  if has_work (HAS_QUOTE)
      +100  for high quality_score
      -length (prefer slightly shorter, cleaner versions once ties are broken)
    """
    def rank_score(q: Dict) -> Tuple[int, int]:
        score = 0
        if q.get("has_person"): score += 1000
        if q.get("has_work"):   score += 500
        score += (q.get("quality_score") or 0) * 10
        
        # Tie-breaker: prioritize cleaner length (penalty for excessive length)
        len_penalty = min(abs(len(q["text"]) - 100), 500)
        return (score, -len_penalty)

    # Sort descending by rank
    cluster_sorted = sorted(cluster, key=rank_score, reverse=True)
    return cluster_sorted[0]


# ─────────────────────────────────────────────
#  Pipeline Execution
# ─────────────────────────────────────────────

def fetch_all_quotes(client: Neo4jClient) -> List[Dict]:
    """Fetch all quotes with their graph relationships for accurate ranking."""
    logger.info("Fetching all quotes and graph state...")
    all_quotes = []
    last_id = -1
    batch_size = 50_000

    while True:
        batch = client.execute_query(
            """
            MATCH (q:Quote)
            WHERE id(q) > $last_id
            WITH q, 
                 exists((:Person)-[:SAID]->(q)) AS has_person,
                 exists((:Work)-[:HAS_QUOTE]->(q)) AS has_work
            RETURN
                id(q) AS node_id,
                q.text AS text,
                q.quality_score AS quality_score,
                has_person,
                has_work
            ORDER BY id(q)
            LIMIT $limit
            """,
            {"last_id": last_id, "limit": batch_size}
        )
        if not batch:
            break
        all_quotes.extend(batch)
        last_id = batch[-1]["node_id"]
        logger.info(f"  Fetched {len(all_quotes):,} quotes...")

    logger.info(f"Total quotes loaded: {len(all_quotes):,}")
    return all_quotes


def run_pipeline(quotes: List[Dict], threshold: float = 0.90) -> List[Dict]:
    """
    Run the 3-stage pipeline in memory.
    Returns a list of dicts describing the updates to be written.
    """
    updates = []
    global_cluster_id = 1
    
    # ── Stage 1: Exact Duplicates ─────────────────────────────────
    logger.info("\n--- STAGE 1: Exact Duplicate Detection ---")
    exact_groups = defaultdict(list)
    
    for q in quotes:
        norm = normalize_text(q["text"])
        q["normalized"] = norm
        q["tokens"] = _tokens(norm)
        if len(norm) >= 15: # Ignore trivial exact matches like "Yes."
            h = exact_hash(norm)
            exact_groups[h].append(q)
            q["exact_hash"] = h
    
    stage1_survivors = []
    exact_clusters_found = 0
    exact_dups_flagged = 0

    for h, group in exact_groups.items():
        if len(group) > 1:
            canonical = select_canonical(group)
            canonical["is_canonical"] = True
            canonical["duplicate_type"] = None
            canonical["duplicate_group_id"] = f"exact_{global_cluster_id}"
            
            stage1_survivors.append(canonical)
            exact_clusters_found += 1
            
            for members in group:
                if members["node_id"] == canonical["node_id"]:
                    continue
                members["is_canonical"] = False
                members["canonical_quote_id"] = canonical["node_id"]
                members["duplicate_type"] = "exact"
                members["duplicate_group_id"] = f"exact_{global_cluster_id}"
                exact_dups_flagged += 1
            
            global_cluster_id += 1
        else:
            stage1_survivors.append(group[0])

    logger.info(f"Exact clusters:  {exact_clusters_found:,}")
    logger.info(f"Exact dups:      {exact_dups_flagged:,}")

    # ── Stage 2: Near Duplicates ──────────────────────────────────
    logger.info("\n--- STAGE 2: Scalable Near-Duplicate Detection ---")
    
    blocks = defaultdict(list)
    for q in stage1_survivors:
        if q.get("is_canonical") is False:
            continue # Already deduplicated
        keys = generate_blocking_keys(q["normalized"], q["tokens"])
        for k in keys:
            blocks[k].append(q)

    # Union-Find for transitive closure of near-duplicates
    parent = {q["node_id"]: q["node_id"] for q in stage1_survivors}
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    pairs_checked = set()
    near_dups_found = 0

    for block_key, block_quotes in blocks.items():
        if len(block_quotes) < 2 or len(block_quotes) > 500:
            continue # Skip noise blocks (e.g. stopword blocks)

        for i in range(len(block_quotes)):
            for j in range(i + 1, len(block_quotes)):
                qa = block_quotes[i]
                qb = block_quotes[j]
                
                # Fast path check
                if find(qa["node_id"]) == find(qb["node_id"]):
                    continue
                    
                pair_id = tuple(sorted([qa["node_id"], qb["node_id"]]))
                if pair_id in pairs_checked:
                    continue
                pairs_checked.add(pair_id)

                # Similarity calculation
                jaccard = _jaccard(set(qa["tokens"]), set(qb["tokens"]))
                if jaccard < (threshold - 0.15): # Fast fail
                    continue
                    
                levenshtein = _norm_levenshtein(qa["normalized"], qb["normalized"])
                
                # Combine signals
                if levenshtein >= threshold or jaccard >= threshold:
                    union(qa["node_id"], qb["node_id"])
                    near_dups_found += 1

    logger.info(f"Pairs compared:  {len(pairs_checked):,}")
    logger.info(f"Near dups found: {near_dups_found:,}")

    # ── Stage 3: Resolve Clusters ─────────────────────────────────
    logger.info("\n--- STAGE 3: Canonical Selection ---")
    
    final_clusters = defaultdict(list)
    id_to_quote = {q["node_id"]: q for q in quotes}
    
    # Map Stage1 survivors into their final Stage2 union-find roots
    for q in stage1_survivors:
        root = find(q["node_id"])
        final_clusters[root].append(q)

    near_clusters_found = 0
    near_dups_flagged = 0

    for root, group in final_clusters.items():
        if len(group) > 1:
            canonical = select_canonical(group)
            cluster_name = f"near_{global_cluster_id}"
            
            canonical["is_canonical"] = True
            canonical["duplicate_type"] = None
            canonical["duplicate_group_id"] = cluster_name
            
            near_clusters_found += 1
            
            for members in group:
                if members["node_id"] == canonical["node_id"]:
                    continue
                # If they were already an exact dup, leave their type as exact,
                # but update their canonical pointer to the top of the food chain.
                if members.get("duplicate_type") != "exact":
                    members["duplicate_type"] = "near"
                
                members["is_canonical"] = False
                members["canonical_quote_id"] = canonical["node_id"]
                members["duplicate_group_id"] = cluster_name
                
                if members["duplicate_type"] == "near":
                    near_dups_flagged += 1
            
            global_cluster_id += 1

    # Format the update batch
    for q in quotes:
        if q.get("duplicate_group_id"):
            updates.append({
                "node_id": q["node_id"],
                "is_canonical": q.get("is_canonical", False),
                "canonical_quote_id": q.get("canonical_quote_id"),
                "duplicate_type": q.get("duplicate_type"),
                "duplicate_group_id": q.get("duplicate_group_id"),
                "quote_hash": q.get("exact_hash")
            })

    total_flagged = exact_dups_flagged + near_dups_flagged
    
    logger.info("\n============================================================")
    logger.info("  DEDUPLICATION SUMMARY")
    logger.info("============================================================")
    logger.info(f"Quotes scanned:               {len(quotes):,}")
    logger.info(f"Exact clusters formed:        {exact_clusters_found:,}")
    logger.info(f"Exact duplicates flagged:     {exact_dups_flagged:,}")
    logger.info(f"Near clusters formed:         {near_clusters_found:,}")
    logger.info(f"Near duplicates flagged:      {near_dups_flagged:,}")
    logger.info(f"Total canonical chosen:       {exact_clusters_found + near_clusters_found:,}")
    logger.info(f"Total quotes to hide:         {total_flagged:,}  ({(total_flagged/max(1, len(quotes))*100):.1f}%)")
    logger.info("============================================================\n")

    return updates


# ─────────────────────────────────────────────
#  Apply to Database
# ─────────────────────────────────────────────

def apply_updates(client: Neo4jClient, updates: List[Dict], batch_size: int = 10_000):
    """Write deduplication flags back to Neo4j."""
    logger.info(f"Writing deduplication flags to Neo4j ({len(updates):,} nodes)...")
    total = len(updates)
    written = 0

    for i in range(0, total, batch_size):
        batch = updates[i : i + batch_size]
        
        client.execute_query(
            """
            UNWIND $rows AS row
            MATCH (q:Quote) WHERE id(q) = row.node_id
            SET q.is_canonical = row.is_canonical,
                q.canonical_quote_id = row.canonical_quote_id,
                q.duplicate_type = row.duplicate_type,
                q.duplicate_group_id = row.duplicate_group_id,
                q.quote_hash = row.quote_hash
            """,
            {"rows": batch}
        )
        written += len(batch)
        logger.info(f"  Written {written:,}/{total:,}")

    logger.info("Writing complete. Database updated safely.")


# ─────────────────────────────────────────────
#  Main Execution
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scalable, Non-Destructive Deduplication")
    parser.add_argument("--apply", action="store_true", help="Write changes to DB")
    parser.add_argument("--limit", type=int, help="Limit quotes fetched for rapid testing")
    parser.add_argument("--threshold", type=float, default=0.90, help="Similarity threshold")
    args = parser.parse_args()

    mode = "APPLY MODE" if args.apply else "DRY RUN"
    logger.info("=" * 60)
    logger.info(f"  Track 2 — Non-Destructive Deduplication Pipeline  [{mode}]")
    logger.info("=" * 60)

    t0 = time.time()
    client = Neo4jClient(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD, config.NEO4J_DATABASE)

    try:
        # 1. Fetch
        quotes = fetch_all_quotes(client)
        if args.limit:
            quotes = quotes[:args.limit]
            logger.info(f"Limiting to {args.limit} quotes for testing.")

        if not quotes:
            return

        # 2. Compute
        updates = run_pipeline(quotes, threshold=args.threshold)

        # 3. Apply
        if args.apply:
            apply_updates(client, updates)
        else:
            logger.info("DRY RUN ONLY. No data was modified. Use --apply to write.")
            logger.info("\nCypher validation queries after apply:")
            logger.info("  // Inspect exact clusters")
            logger.info("  MATCH (q:Quote) WHERE q.duplicate_type = 'exact' RETURN q.duplicate_group_id, count(q) ORDER BY count(q) DESC LIMIT 10")
            logger.info("  // Compare canonical vs duplicate")
            logger.info("  MATCH (dup:Quote {is_canonical: false})")
            logger.info("  MATCH (canon:Quote) WHERE id(canon) = dup.canonical_quote_id")
            logger.info("  RETURN canon.text, dup.text LIMIT 20")

    finally:
        client.close()
        logger.info(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
