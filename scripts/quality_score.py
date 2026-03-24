"""
Track 4 — Quality Scoring Pipeline
=====================================

Assigns a deterministic quality_score, bucket, and flags to every Quote node.
Safe to run multiple times (idempotent updates).
"""

import argparse
import logging
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.neo4j_client import Neo4jClient
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Quality signals 
# ─────────────────────────────────────────────

_WIKI_JUNK = re.compile(r'\{\{|\[\[|<ref|</ref|\|\s*\w|\bFile:|\bImage:|==\s*\w|&[a-z]+;|&#\d+;|\bCategory:', re.IGNORECASE)
_METADATA_LEAK = re.compile(r'(?:p\.\s*\d+|pp\.\s*\d+|\[citation needed\]|\[[0-9]+\]|Read more|Continue reading)', re.IGNORECASE)
_SUSPICIOUS_FORMATTING = re.compile(r'[.\-_]{5,}|^\s*\*|(?:(?:^|\s)[a-z]){5,}') # excessive punct or weird spacing

SCORING_VERSION = "v2_strict"

def score_quote(
    text: str,
    has_person: bool,
    has_work: bool,
    has_source: bool,
    is_non_canonical: bool,
    needs_review: bool
) -> Tuple[int, str, bool, List[str]]:
    """Returns (score, bucket, high_quality, reasons)"""
    score = 0
    reasons = []
    
    text = text or ""
    length = len(text.strip())

    # ── Positive signals ─────────────────────────────────────────
    if has_person:
        score += 2
        reasons.append("+2:has_person")
    
    if has_work:
        score += 2
        reasons.append("+2:has_work")
        
    if has_source:
        score += 1
        reasons.append("+1:has_source")

    if 30 <= length <= 300:
        score += 1
        reasons.append("+1:good_length")

    if not _WIKI_JUNK.search(text) and not _METADATA_LEAK.search(text):
        score += 1
        reasons.append("+1:clean_text")

    # ── Negative signals ─────────────────────────────────────────
    if is_non_canonical:
        score -= 3
        reasons.append("-3:duplicate")

    if _METADATA_LEAK.search(text) or _WIKI_JUNK.search(text):
        score -= 2
        reasons.append("-2:metadata_leak")

    if length < 30:
        score -= 2
        reasons.append("-2:too_short")
    elif length > 600:
        score -= 2
        reasons.append("-2:too_long")

    if needs_review:
        score -= 3
        reasons.append("-3:needs_review")

    if _SUSPICIOUS_FORMATTING.search(text):
        score -= 1
        reasons.append("-1:suspicious_formatting")

    # ── Bucketing ────────────────────────────────────────────────
    if score >= 5:
        bucket = "high_quality"
    elif 3 <= score <= 4:
        bucket = "review"
    elif 1 <= score <= 2:
        bucket = "low_quality"
    else:
        bucket = "garbage"
        
    high_quality_flag = (bucket == "high_quality")

    return score, bucket, high_quality_flag, reasons


# ─────────────────────────────────────────────
#  Database Operations
# ─────────────────────────────────────────────

def fetch_quote_graph_state(client: Neo4jClient, batch_size: int = 10_000) -> List[Dict]:
    logger.info("Fetching quote graph state from Neo4j...")
    results = []
    last_id = -1
    
    while True:
        batch = client.execute_query(
            """
            MATCH (q:Quote)
            WHERE id(q) > $last_id
            WITH q,
                 exists((:Person)-[:SAID]->(q)) AS has_person,
                 exists((:Work)-[:HAS_QUOTE]->(q)) AS has_work,
                 exists((:Source)-[:HAS_QUOTE]->(q)) AS has_source
            RETURN
                id(q)             AS node_id,
                coalesce(q.text_before_cleaning, q.text) AS original_text_fallback,
                q.text            AS text,
                q.is_canonical    AS is_canonical,
                q.needs_review    AS needs_review,
                has_person,
                has_work,
                has_source
            ORDER BY id(q)
            LIMIT $limit
            """,
            {"last_id": last_id, "limit": batch_size},
        )
        if not batch:
            break
            
        results.extend(batch)
        last_id = batch[-1]["node_id"]
        logger.info(f"  Fetched {len(results):,} quotes...")
        
    logger.info(f"Total quotes fetched: {len(results):,}")
    return results


def write_scores(client: Neo4jClient, scored: List[Dict], batch_size: int = 10_000, dry_run: bool = False):
    if dry_run:
        logger.info("DRY RUN — skipping writes to Neo4j")
        return

    logger.info(f"Writing scores to Neo4j ({len(scored):,} quotes)...")
    total = len(scored)
    written = 0
    now = datetime.utcnow().isoformat()

    for i in range(0, total, batch_size):
        batch = scored[i : i + batch_size]
        
        params = [
            {
                "node_id":         r["node_id"],
                "score":           r["score"],
                "bucket":          r["bucket"],
                "high_quality":    r["high_quality"],
                "needs_review":    r["original_needs_review"], # preserving original review flag if it existed
                "scored_at":       now,
                "scoring_version": SCORING_VERSION
            }
            for r in batch
        ]

        client.execute_query(
            """
            UNWIND $rows AS row
            MATCH (q:Quote) WHERE id(q) = row.node_id
            SET q.quality_score   = row.score,
                q.quality_bucket  = row.bucket,
                q.high_quality    = row.high_quality,
                q.scored_at       = row.scored_at,
                q.scoring_version = row.scoring_version
            """,
            {"rows": params},
        )
        written += len(batch)
        logger.info(f"  Written: {written:,}/{total:,}")

    logger.info("Write complete.")


# ─────────────────────────────────────────────
#  Reporting
# ─────────────────────────────────────────────

def print_report(scored: List[Dict]):
    bucket_counts = Counter(r["bucket"] for r in scored)
    logger.info("=" * 60)
    logger.info("QUALITY DISTRIBUTION")
    logger.info("=" * 60)
    logger.info(f"Total scored: {len(scored):,}")
    logger.info(f"  High Quality (>= 5): {bucket_counts['high_quality']:,}")
    logger.info(f"  Review       (3-4):  {bucket_counts['review']:,}")
    logger.info(f"  Low Quality  (1-2):  {bucket_counts['low_quality']:,}")
    logger.info(f"  Garbage      (<=0):  {bucket_counts['garbage']:,}")
    
    logger.info("\n--- 50 SAMPLES EXPLAINED ---")
    for idx, r in enumerate(scored[:50]):
        text = r["text"].replace("\n", " ")[:100]
        logger.info(f"[{r['score']:>2}] {r['bucket']:<12} | {', '.join(r['reasons'])}")
        logger.info(f"     \"{text}...\"")
        logger.info("-" * 40)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write changes to DB")
    args = parser.parse_args()

    mode = "APPLY MODE" if args.apply else "DRY RUN"
    logger.info("=" * 60)
    logger.info(f"  Track 4 — Quality Scoring  [{mode}]")
    logger.info("=" * 60)

    client = Neo4jClient(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD, config.NEO4J_DATABASE)
    try:
        quotes = fetch_quote_graph_state(client)
        scored = []
        
        logger.info("Computing quality scores...")
        for q in quotes:
            text = q.get("text") or q.get("original_text_fallback") or ""
            is_non_canonical = q.get("is_canonical") is False
            needs_review = bool(q.get("needs_review"))
            
            s, b, hq, reasons = score_quote(
                text=text,
                has_person=q.get("has_person"),
                has_work=q.get("has_work"),
                has_source=q.get("has_source"),
                is_non_canonical=is_non_canonical,
                needs_review=needs_review
            )
            
            scored.append({
                "node_id": q["node_id"],
                "text": text,
                "score": s,
                "bucket": b,
                "high_quality": hq,
                "reasons": reasons,
                "original_needs_review": needs_review
            })
            
        print_report(scored)
        write_scores(client, scored, dry_run=not args.apply)

        if args.apply:
            logger.info("\nValidation queries available in script output.")

    finally:
        client.close()

if __name__ == "__main__":
    main()
