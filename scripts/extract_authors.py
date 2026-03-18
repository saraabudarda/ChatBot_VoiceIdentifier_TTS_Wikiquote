"""
Author Extraction Pipeline for Orphan Quotes
==============================================

Context:
Many quotes are orphans (missing SAID relationships) because the author
name is embedded inside the Quote.text.
Example: "Be the change you wish to see." — Mahatma Gandhi

Goal:
Detect these attributions, extract the author name, clean the quote text,
reuse existing Person nodes if possible, create new Person nodes if needed,
and link them via SAID relationships.

Strict Requirements:
1. ONLY process quotes with NO incoming SAID relationships.
2. High precision > High recall. Only extract if highly confident.
3. Reject garbage authors (URLs, dates, source metadata, too long).
4. Save audit fields (author_extracted, extraction_confidence, text_before_cleaning).
5. Always support --dry-run first.
"""

import argparse
import csv
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.neo4j_client import Neo4jClient
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Strict Extraction Rules & Regex
# ─────────────────────────────────────────────

# Matches attributions at the END of the string.
# Supports: —, -, ~, or wrapped in parentheses
# Limits author name to 5-50 chars to avoid capturing whole sentences.
_ATTRIBUTION_PATTERN = re.compile(
    r'(?P<quote>.*?)'                                      # The actual quote
    r'(?:\s+)'                                              # Required whitespace separation
    r'(?:'                                                  # Start of attribution block
        r'[\u2014\u2013\-~]\s*(?P<author1>[A-Z][^\n]{4,50}?)'  # — Author
        r'|'                                                # OR
        r'\(\s*(?P<author2>[A-Z][^\n]{4,50}?)\s*\)'           # (Author)
    r')\s*$',                                               # End of string
    re.DOTALL
)

# Strings that look like authors but are actually metadata or garbage
_BLACKLIST_LOWER = frozenset({
    "unknown", "anonymous", "read more", "continue reading",
    "wikipedia", "wikiquote", "internet movie database", "imdb",
    "the new york times", "new york times", "the guardian", "bbc",
    "chapter", "page", "ibid", "op cit", "et al", "translated by",
    "edited by", "directed by", "youtube", "twitter", "facebook"
})

_BLACKLIST_PREFIXES = ("page ", "ch ", "chapter ", "vol ", "volume ", "http", "www", "read ", "from ")


def _normalize_author_key(name: str) -> str:
    """Canonical key for exact matching against existing Person nodes."""
    # Remove punctuation except hyphens and apostrophes, lower, strip, collapse spaces
    s = re.sub(r'[^\w\s\-\']', '', name.lower())
    return re.sub(r'\s+', ' ', s).strip()


def validate_author_candidate(name: str) -> Tuple[bool, str]:
    """
    Strict validation of the extracted string to ensure it's likely a person.
    Returns (is_valid, reason_if_invalid)
    """
    clean_name = re.sub(r'\s+', ' ', name).strip()
    lower_name = clean_name.lower()

    if len(clean_name) < 4:
        return False, "Too short"
    if len(clean_name) > 60:
        return False, "Too long"

    # Reject if it contains URLs
    if "http" in lower_name or "www." in lower_name or ".com" in lower_name:
        return False, "Contains URL"

    # Reject if it has too many digits (likely a date, page number, or verse)
    digit_count = sum(c.isdigit() for c in clean_name)
    if digit_count >= 4:
        return False, "Too many digits (likely date/metadata)"

    # Reject common non-author metadata
    if lower_name in _BLACKLIST_LOWER:
        return False, f"Blacklisted exact match: {lower_name}"

    for prefix in _BLACKLIST_PREFIXES:
        if lower_name.startswith(prefix):
            return False, f"Blacklisted prefix: {prefix}"

    # Reject if it doesn't look like a capitalized name (must have at least one capital letter)
    if not any(c.isupper() for c in clean_name):
         return False, "No capital letters"

    return True, ""


def extract_author(text: str) -> Optional[Tuple[str, str, float, str]]:
    """
    Attempt to extract an author from the end of a quote.
    Returns (cleaned_quote, author_name, confidence, pattern) or None.
    """
    text = text.strip()
    if not text:
        return None

    match = _ATTRIBUTION_PATTERN.search(text)
    if not match:
        return None

    quote_part = match.group('quote').strip()
    author_part = (match.group('author1') or match.group('author2')).strip()

    # Clean up any trailing punctuation on author (like a stray period)
    author_part = re.sub(r'[.,;:]$', '', author_part).strip()

    # Strip surrounding quotes from the quote text if they exist
    quote_part = re.sub(r'^["\'\u201c\u2018]+|["\'\u201d\u2019]+$', '', quote_part).strip()

    is_valid, reject_reason = validate_author_candidate(author_part)

    if not is_valid:
        # We found a pattern, but it's garbage.
        return None

    # Confidence scoring
    confidence = 0.90
    # Higher confidence if it's a known format like "First Last"
    words = author_part.split()
    if 2 <= len(words) <= 3 and all(w[0].isupper() for w in words if w.isalpha()):
        confidence = 0.98

    pattern_used = "dash" if match.group('author1') else "parens"

    return quote_part, author_part, confidence, pattern_used


# ─────────────────────────────────────────────
#  Database Operations
# ─────────────────────────────────────────────

def fetch_existing_persons(client: Neo4jClient) -> Dict[str, str]:
    """
    Load all existing Person nodes to memory for fast matching.
    Returns dict mapping normalized_key -> exact_db_name.
    """
    logger.info("Loading existing Person nodes for matching...")
    results = client.execute_query("MATCH (p:Person) RETURN p.name AS name")
    mapping = {}
    for r in results:
        name = r["name"]
        key = _normalize_author_key(name)
        if key:
            mapping[key] = name
    logger.info(f"Loaded {len(mapping):,} unique Person keys.")
    return mapping


def fetch_orphan_quotes(client: Neo4jClient, batch_size: int = 50_000) -> List[Dict]:
    """
    Fetch quotes that DO NOT have an incoming SAID relationship from a Person.
    Returns: List of dicts {node_id, text}
    """
    logger.info("Fetching orphan quotes (no SAID relationship)...")
    all_quotes = []
    last_id = -1

    while True:
        batch = client.execute_query(
            """
            MATCH (q:Quote)
            WHERE id(q) > $last_id
              AND NOT (:Person)-[:SAID]->(q)
              AND coalesce(q.author_extracted, false) = false
            RETURN id(q) AS node_id, q.text AS text
            ORDER BY id(q)
            LIMIT $limit
            """,
            {"last_id": last_id, "limit": batch_size}
        )
        if not batch:
            break

        all_quotes.extend(batch)
        last_id = batch[-1]["node_id"]
        logger.info(f"  Fetched {len(all_quotes):,} orphans...")

    logger.info(f"Total orphan quotes found: {len(all_quotes):,}")
    return all_quotes


def apply_extractions(client: Neo4jClient, successful_extractions: List[Dict], batch_size: int = 5000):
    """
    Apply the accepted extractions to the database.
    Updates the Quote text, adds audit fields, creates/merges the Person, and links them.
    Idempotent due to MERGE.
    """
    logger.info(f"Writing {len(successful_extractions):,} extractions to Neo4j...")
    total = len(successful_extractions)
    written = 0

    for i in range(0, total, batch_size):
        batch = successful_extractions[i : i + batch_size]

        client.execute_query(
            """
            UNWIND $rows AS row

            // 1. Get the quote
            MATCH (q:Quote) WHERE id(q) = row.node_id

            // 2. Audit fields and update text
            SET q.text_before_extraction = row.original_text,
                q.text = row.clean_text,
                q.author_extracted = true,
                q.extraction_confidence = row.confidence,
                q.extraction_pattern = row.pattern

            // 3. Create or Match the Person (Idempotent)
            WITH q, row
            MERGE (p:Person {name: row.final_author_name})

            // 4. Create the relationship (Idempotent)
            MERGE (p)-[:SAID]->(q)
            """,
            {"rows": batch}
        )
        written += len(batch)
        logger.info(f"  Written {written:,}/{total:,}")

    logger.info("Write complete.")


# ─────────────────────────────────────────────
#  Main Execution
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract embedded authors from orphan quotes.")
    parser.add_argument("--apply", action="store_true", help="Write changes to DB (default: Dry Run)")
    parser.add_argument("--limit", type=int, help="Limit quotes processed for quick testing")
    args = parser.parse_args()

    mode = "APPLY MODE" if args.apply else "DRY RUN"
    logger.info("=" * 60)
    logger.info(f"  Author Extraction Pipeline  [{mode}]")
    logger.info("=" * 60)

    client = Neo4jClient(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD, config.NEO4J_DATABASE)

    try:
        # 1. Load existing authors for resolution
        existing_persons = fetch_existing_persons(client)

        # 2. Fetch orphans
        orphans = fetch_orphan_quotes(client)
        if args.limit:
            orphans = orphans[:args.limit]
            logger.info(f"Limited to {args.limit} orphans for testing.")

        if not orphans:
            logger.info("No orphan quotes found. Exiting.")
            return

        # 3. Process
        logger.info("Extracting authors...")
        successes = []
        author_counts = Counter()
        reused_count = 0
        new_count = 0
        samples_to_print = []

        for q in orphans:
            result = extract_author(q["text"])
            if not result:
                continue

            clean_quote, raw_author, conf, pattern = result

            # Resolve author name
            norm_key = _normalize_author_key(raw_author)
            if norm_key in existing_persons:
                final_author = existing_persons[norm_key]
                reused_count += 1
            else:
                final_author = raw_author
                new_count += 1

            author_counts[final_author] += 1

            extract_data = {
                "node_id": q["node_id"],
                "original_text": q["text"],
                "clean_text": clean_quote,
                "raw_author_extracted": raw_author,
                "final_author_name": final_author,
                "confidence": conf,
                "pattern": pattern,
                "is_reused": norm_key in existing_persons
            }
            successes.append(extract_data)

            if len(samples_to_print) < 50:
                samples_to_print.append(extract_data)

        # 4. Report
        logger.info("=" * 60)
        logger.info("EXTRACTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Quotes scanned:    {len(orphans):,}")
        logger.info(f"Quotes matched:    {len(successes):,} ({(len(successes)/len(orphans)*100 if orphans else 0):.1f}%)")
        logger.info(f"Authors reused:    {reused_count:,}")
        logger.info(f"New authors found: {new_count:,}")

        logger.info("\nTop 10 Extracted Authors:")
        for name, count in author_counts.most_common(10):
            logger.info(f"  {name:<30} {count:,}")

        if not args.apply:
            logger.info("\nDry Run: Sample Transformations (First 50)")
            logger.info("-" * 60)
            for s in samples_to_print:
                status = "[REUSED]" if s["is_reused"] else "[ NEW  ]"
                print(f"RAW : {s['original_text']}")
                print(f"TEXT: {s['clean_text']}")
                print(f"AUTH: {status} {s['final_author_name']}  (conf: {s['confidence']})")
                print("-" * 40)
            logger.info("\nDRY RUN ONLY. Use --apply to write to database.")
        else:
            if successes:
                apply_extractions(client, successes)
            else:
                logger.info("No successful extractions to apply.")

    finally:
        client.close()

if __name__ == "__main__":
    main()
