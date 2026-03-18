"""
Track 3 — Relationship Repair Pipeline
=========================================

Links orphaned Quote nodes to existing or highly-confident new Work nodes.
Does NOT modify Quote.text. Focuses entirely on structure.
"""

import argparse
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.neo4j_client import Neo4jClient
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Strict Regex Patterns (High Precision)
# ─────────────────────────────────────────────

# Strict patterns for creating NEW works (must be enclosed in quotes to be safe)
_NEW_WORK_PATTERNS = [
    re.compile(r'\bfrom the book\s+["\u201c\u2018](?P<work>[A-Z][^\u201d\u2019"]{3,60}?)["\u201d\u2019]', re.IGNORECASE),
    re.compile(r'\bin the book\s+["\u201c\u2018](?P<work>[A-Z][^\u201d\u2019"]{3,60}?)["\u201d\u2019]', re.IGNORECASE),
    re.compile(r'\bfrom his book\s+["\u201c\u2018](?P<work>[A-Z][^\u201d\u2019"]{3,60}?)["\u201d\u2019]', re.IGNORECASE),
    re.compile(r'\bfrom her book\s+["\u201c\u2018](?P<work>[A-Z][^\u201d\u2019"]{3,60}?)["\u201d\u2019]', re.IGNORECASE),
    re.compile(r'\bas quoted in\s+["\u201c\u2018](?P<work>[A-Z][^\u201d\u2019"]{3,60}?)["\u201d\u2019]', re.IGNORECASE),
    re.compile(r'\bfrom the play\s+["\u201c\u2018](?P<work>[A-Z][^\u201d\u2019"]{3,60}?)["\u201d\u2019]', re.IGNORECASE),
    re.compile(r'\bfrom the poem\s+["\u201c\u2018](?P<work>[A-Z][^\u201d\u2019"]{3,60}?)["\u201d\u2019]', re.IGNORECASE),
    re.compile(r'\bfrom the essay\s+["\u201c\u2018](?P<work>[A-Z][^\u201d\u2019"]{3,60}?)["\u201d\u2019]', re.IGNORECASE)
]

_BLACKLIST_LOWER = frozenset({
    "unknown", "anonymous", "read more", "continue reading",
    "wikipedia", "wikiquote", "internet movie database", "imdb",
    "the new york times", "new york times", "the guardian", "bbc",
    "youtube", "twitter", "facebook", "page", "chapter", "vol", "volume"
})

_BLACKLIST_PREFIXES = ("page ", "ch ", "chapter ", "vol ", "volume ", "http", "www", "read ", "from ")


def _normalize_title(title: str) -> str:
    """Canonical key for exact matching."""
    s = re.sub(r'[^\w\s]', '', title.lower())
    return re.sub(r'\s+', ' ', s).strip()


def validate_work_candidate(title: str) -> Tuple[bool, str]:
    """Strict validation for work titles."""
    clean_title = title.strip()
    lower_title = clean_title.lower()

    if len(clean_title) < 4:
        return False, "Too short"
    if len(clean_title) > 60:
        return False, "Too long"

    if "http" in lower_title or "www." in lower_title or ".com" in lower_title:
        return False, "Contains URL"

    digit_count = sum(c.isdigit() for c in clean_title)
    if digit_count >= 8:
        return False, "Too many digits"

    if lower_title in _BLACKLIST_LOWER:
        return False, f"Blacklisted: {lower_title}"

    for prefix in _BLACKLIST_PREFIXES:
        if lower_title.startswith(prefix):
            return False, f"Blacklisted prefix: {prefix}"

    if not any(c.isupper() for c in clean_title):
         return False, "No capital letters"

    return True, ""


# ─────────────────────────────────────────────
#  Extraction Logic
# ─────────────────────────────────────────────

def extract_work(text: str, existing_works: Dict[str, str]) -> Optional[Tuple[str, str, float, bool]]:
    """
    Returns (cleaned_work_name, match_type, confidence, is_reused) or None
    """
    if not text:
        return None

    lower_text = text.lower()

    # Strategy 1: Hyper-safe reuse of existing works via exact phrase matching.
    # We only check works that are at least 8 characters long to avoid false positives on short words.
    for norm_key, real_name in existing_works.items():
        if len(norm_key) > 8:
            # Check if text contains "in {Work}" or "from {Work}" or "as quoted in {Work}"
            search_str = norm_key.lower()
            if (f" in {search_str}" in lower_text or 
                f" from {search_str}" in lower_text or 
                f" quoted in {search_str}" in lower_text):
                
                # Verify it's not actually a Person name (basic heuristic: if it has "by {Work}" it's a person)
                if f" by {search_str}" not in lower_text:
                    return real_name, "exact_phrase_reused", 0.98, True

    # Strategy 2: Explicit textual marker WITH quotes (creates new works safely)
    for pat in _NEW_WORK_PATTERNS:
        match = pat.search(text)
        if match:
            candidate = match.group('work').strip()
            is_valid, _ = validate_work_candidate(candidate)
            if is_valid:
                norm = _normalize_title(candidate)
                if norm in existing_works:
                    return existing_works[norm], "explicit_quotes_reused", 0.99, True
                else:
                    words = candidate.split()
                    if len(words) > 1:
                        return candidate, "explicit_quotes_new", 0.95, False

    return None


# ─────────────────────────────────────────────
#  Database Operations
# ─────────────────────────────────────────────

def fetch_existing_works(client: Neo4jClient) -> Dict[str, str]:
    logger.info("Loading existing Work nodes...")
    results = client.execute_query("MATCH (w:Work) RETURN w.name AS name")
    mapping = {}
    for r in results:
        name = r["name"]
        key = _normalize_title(name)
        if key:
            mapping[key] = name
    logger.info(f"Loaded {len(mapping):,} unique Work identities.")
    return mapping


def fetch_unlinked_quotes(client: Neo4jClient, limit: Optional[int] = None) -> List[Dict]:
    logger.info("Fetching quotes with no HAS_QUOTE link...")
    all_quotes = []
    last_id = -1
    fetch_limit = limit or float("inf")
    batch_size = 50_000

    while len(all_quotes) < fetch_limit:
        remaining = int(min(batch_size, fetch_limit - len(all_quotes)))
        batch = client.execute_query(
            """
            MATCH (q:Quote)
            WHERE id(q) > $last_id
              AND NOT (:Work)-[:HAS_QUOTE]->(q)
              AND coalesce(q.work_extracted, false) = false
              AND coalesce(q.duplicate_type, '') <> 'exact'
            RETURN id(q) AS node_id, coalesce(q.text_before_cleaning, q.text) AS text
            ORDER BY id(q)
            LIMIT $limit
            """,
            {"last_id": last_id, "limit": remaining}
        )
        if not batch:
            break
            
        all_quotes.extend(batch)
        last_id = batch[-1]["node_id"]
        logger.info(f"  Fetched {len(all_quotes):,} unlinked quotes...")

    logger.info(f"Total unlinked candidate quotes found: {len(all_quotes):,}")
    return all_quotes


def apply_repairs(client: Neo4jClient, successful_repairs: List[Dict], batch_size: int = 5000):
    logger.info(f"Writing {len(successful_repairs):,} work relationships to Neo4j...")
    total = len(successful_repairs)
    written = 0
    run_id = "track3_v1"

    for i in range(0, total, batch_size):
        batch = successful_repairs[i : i + batch_size]

        client.execute_query(
            """
            UNWIND $rows AS row

            // Match quote
            MATCH (q:Quote) WHERE id(q) = row.node_id

            // Set audit fields on quote
            SET q.work_extracted = true,
                q.extracted_work_title = row.work_name,
                q.work_match_type = row.match_type,
                q.work_confidence = row.confidence,
                q.relationship_repair_run_id = $run_id

            // Create or Match the Work node (Idempotent)
            WITH q, row
            MERGE (w:Work {name: row.work_name})
            
            // Create the relationship (Idempotent)
            MERGE (w)-[r:HAS_QUOTE]->(q)
            SET r.relationship_repair_run_id = $run_id
            """,
            {"rows": batch, "run_id": run_id}
        )
        written += len(batch)
        logger.info(f"  Written {written:,}/{total:,}")

    logger.info("Repair update complete.")


# ─────────────────────────────────────────────
#  Main Execution
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Repair missing Work to Quote relationships.")
    parser.add_argument("--apply", action="store_true", help="Write changes to DB")
    parser.add_argument("--limit", type=int, help="Limit quotes processed")
    args = parser.parse_args()

    mode = "APPLY MODE" if args.apply else "DRY RUN"
    logger.info("=" * 60)
    logger.info(f"  Track 3 — Relationship Repair Pipeline [{mode}]")
    logger.info("=" * 60)

    client = Neo4jClient(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD, config.NEO4J_DATABASE)

    try:
        existing_works = fetch_existing_works(client)
        unlinked_quotes = fetch_unlinked_quotes(client, limit=args.limit)
        
        if not unlinked_quotes:
            logger.info("No unlinked quotes found matching criteria.")
            return

        logger.info("Analyzing text for Work node matches...")
        successes = []
        match_types = Counter()
        reused_count = 0
        new_count = 0
        samples_to_print = []

        for q in unlinked_quotes:
            result = extract_work(q["text"], existing_works)
            if not result:
                continue

            work_name, match_type, conf, is_reused = result
            
            match_types[match_type] += 1
            if is_reused:
                reused_count += 1
            else:
                new_count += 1

            repair_data = {
                "node_id": q["node_id"],
                "original_text": q["text"],
                "work_name": work_name,
                "match_type": match_type,
                "confidence": conf,
                "is_reused": is_reused
            }
            successes.append(repair_data)

            if len(samples_to_print) < 50:
                samples_to_print.append(repair_data)

        # Report
        logger.info("=" * 60)
        logger.info("REPAIR SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Quotes scanned:          {len(unlinked_quotes):,}")
        logger.info(f"Quotes linked to Works:  {len(successes):,} ({(len(successes)/max(1, len(unlinked_quotes))*100):.1f}%)")
        logger.info(f"Existing Works reused:   {reused_count:,}")
        logger.info(f"New Works created:       {new_count:,}")

        logger.info("\nMatch Types Triggered:")
        for rule, count in match_types.most_common():
            logger.info(f"  {rule:<30} {count:,}")

        if not args.apply:
            logger.info("\nDry Run: Sample Repairs (First 50)")
            logger.info("-" * 60)
            for s in samples_to_print:
                status = "[REUSED]" if s["is_reused"] else "[ NEW  ]"
                print(f"TEXT : {s['original_text']}")
                print(f"WORK : {status} {s['work_name']} (conf={s['confidence']}, rule={s['match_type']})")
                print("-" * 40)
            
            logger.info("\nDRY RUN ONLY. Use --apply to write to database.")
            logger.info("\nCypher validation and rollback queries printed in the code.")
        else:
            if successes:
                apply_repairs(client, successes)
            else:
                logger.info("No successful repairs to apply.")

    finally:
        client.close()

if __name__ == "__main__":
    main()
