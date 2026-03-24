"""
Track 1 — Quote Text Cleaning Pipeline
=========================================

Context:
About 13.7% of quotes have metadata accidentally ingested (e.g., "Read more",
"Page 32", "Chapter 4", "[1]"). This script cleans the quoted text safely.

Goal:
Detect and remove UI fragments, book metadata, Wikipedia citation junk, and
HTML leftovers. Never damage the core quote.

Strict Requirements:
1. ONLY clean Quote.text.
2. High precision > High recall.
3. Keep track of what was removed for auditing.
4. If a quote is more than 30% metadata, flag for review rather than auto-clean.
"""

import argparse
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.neo4j_client import Neo4jClient
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Strict Regex Patterns (High Precision)
# ─────────────────────────────────────────────

# 1. UI and Navigation junk (usually at the very end or beginning)
_UI_JUNK = re.compile(
    r'(?:\b(?:Read more|See more|Show more|Continue reading|Share|Copy|Tweet|Pin it)\b[.:\s]*)$'
    r'|^(?:\b(?:Read more|See more|Show more)\b[.:\s]*)',
    re.IGNORECASE
)

# 2. Wikipedia / Citation brackets e.g. [1], [12], [citation needed]
_WIKI_BRACKETS = re.compile(
    r'\[\d+\]'                           # [1] or [123]
    r'|\[(?:citation needed|note \d+|source)\]',  # [citation needed]
    re.IGNORECASE
)

# 3. Source metadata at the END of a quote
# e.g., (Page 35), - Chapter 4, Vol. 2
_SOURCE_METADATA_END = re.compile(
    r'(?:[\(\[\-\—]\s*)?'                                         # Optional opening paren/dash
    r'(?:'
        r'(?:p\.|pp\.|page|pages)\s*\d+(?:-\d+)?'                 # p. 45 or pages 45-46
        r'|(?:vol\.|volume)\s*\d+'                                # Vol. 2
        r'|(?:ch\.|chapter)\s*\w+'                                # Chapter 4 or Chapter IV
        r'|(?:issue|edition)\s*\d+'                               # Issue 5
    r')'
    r'(?:\s*[\)\]])?'                                             # Optional closing paren
    r'\s*$',                                                      # Must be at the end
    re.IGNORECASE
)

# 4. Stray HTML entities or leftover tags
_HTML_JUNK = re.compile(
    r'&[a-z]+;'           # &quot; &nbsp;
    r'|&#\d+;'            # &#160;
    r'|<\/?\w+[^>]*>'     # <b> </b> <br/>
    r'|^\s*\*+\s*'        # Leading bullet points (asterisks)
    r'|^\s*=\s*'          # Leading equal signs
)

# 5. Excessive punctuation
_EXCESSIVE_PUNCT = re.compile(r'\.{4,}')  # 4 or more dots → replace with …


# ─────────────────────────────────────────────
#  Cleaning Logic
# ─────────────────────────────────────────────

def clean_quote(text: str) -> Tuple[str, List[str], float, bool]:
    """
    Applies cleaning rules to the text.
    Returns: (cleaned_text, list_of_rules_applied, confidence, needs_review)
    """
    original = text
    cleaned = text
    rules_applied = []
    
    # 1. HTML & Basic Formatting
    if _HTML_JUNK.search(cleaned):
        cleaned = _HTML_JUNK.sub('', cleaned)
        rules_applied.append('HTML_JUNK')

    # 2. Wikipedia Brackets
    if _WIKI_BRACKETS.search(cleaned):
        cleaned = _WIKI_BRACKETS.sub('', cleaned)
        rules_applied.append('WIKI_BRACKETS')

    # 3. UI Junk (Read more, etc)
    if _UI_JUNK.search(cleaned):
        cleaned = _UI_JUNK.sub('', cleaned)
        rules_applied.append('UI_JUNK')

    # 4. Source Metadata at end (Page 45)
    if _SOURCE_METADATA_END.search(cleaned):
        cleaned = _SOURCE_METADATA_END.sub('', cleaned)
        rules_applied.append('SOURCE_METADATA')

    # 5. Excessive formatting
    if _EXCESSIVE_PUNCT.search(cleaned):
        cleaned = _EXCESSIVE_PUNCT.sub('…', cleaned)
        rules_applied.append('EXCESSIVE_PUNCT')

    # Safe Normalization
    cleaned = cleaned.replace('\xa0', ' ')             # non-breaking spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)             # collapse spaces
    cleaned = re.sub(r'^["\'\u201c\u2018]+|["\'\u201d\u2019]+$', '', cleaned.strip()).strip() # strip wrapping quotes

    # If no real changes (other than maybe one space), return empty
    if original.strip() == cleaned.strip() and not rules_applied:
        return text, [], 0.0, False

    # Calculate how much was removed to determine confidence
    orig_len = len(original.strip())
    new_len = len(cleaned.strip())
    removed_ratio = 1.0 - (new_len / max(1, orig_len))

    confidence = 1.0
    needs_review = False

    if new_len < 10:
        # We cleaned away almost everything. Probably damaged.
        confidence = 0.1
        needs_review = True
    elif removed_ratio > 0.30:
        # We removed more than 30% of the text. Suspicious.
        confidence = 0.5
        needs_review = True
    elif 'UI_JUNK' in rules_applied or 'SOURCE_METADATA' in rules_applied:
        # Obvious junk removal is high confidence
        confidence = 0.95

    return cleaned, rules_applied, confidence, needs_review


# ─────────────────────────────────────────────
#  Database Operations
# ─────────────────────────────────────────────

def fetch_dirty_quotes(client: Neo4jClient, limit: Optional[int] = None) -> List[Dict]:
    """
    Fetch quotes that likely need cleaning.
    Instead of fetching 1.3M, we use the negative signals found in Track 4.
    """
    logger.info("Fetching quotes that contain potential metadata leaks...")
    last_id = -1
    all_quotes = []
    fetch_limit = limit or float("inf")
    batch_size = 50_000

    # The WHERE clause uses regex patterns natively in Neo4j (using =~ '(?i).*pattern.*')
    # or we just fetch quotes that scored poorly on 'clean_text' in the last run.
    # To be safe and comprehensive, we'll fetch quotes that have NOT been marked clean yet.
    
    while len(all_quotes) < fetch_limit:
        remaining = int(min(batch_size, fetch_limit - len(all_quotes)))
        batch = client.execute_query(
            """
            MATCH (q:Quote)
            WHERE id(q) > $last_id
              AND coalesce(q.cleaning_applied, false) = false
              AND (
                  q.text =~ '(?i).*(read more|see more|show more).*' OR
                  q.text =~ '(?i).*(page |p\\.|pp\\.|vol\\.|chapter ).*' OR
                  q.text =~ '.*\\[\\d+\\].*' OR
                  q.text =~ '.*&[a-zA-Z]+;.*' OR
                  q.text =~ '.*<[^>]+>.*'
              )
            RETURN id(q) AS node_id, q.text AS text
            ORDER BY id(q)
            LIMIT $limit
            """,
            {"last_id": last_id, "limit": remaining}
        )
        if not batch:
            break
            
        all_quotes.extend(batch)
        last_id = batch[-1]["node_id"]
        logger.info(f"  Fetched {len(all_quotes):,} dirty candidates...")

    logger.info(f"Total dirty candidates found: {len(all_quotes):,}")
    return all_quotes


def apply_cleaning(client: Neo4jClient, successful_cleans: List[Dict], batch_size: int = 5000):
    """
    Apply the cleaned text to the database.
    Updates the Quote text and adds audit fields.
    """
    logger.info(f"Writing {len(successful_cleans):,} cleaned quotes to Neo4j...")
    total = len(successful_cleans)
    written = 0

    for i in range(0, total, batch_size):
        batch = successful_cleans[i : i + batch_size]

        client.execute_query(
            """
            UNWIND $rows AS row

            MATCH (q:Quote) WHERE id(q) = row.node_id

            // Update text and set audit fields
            SET q.text_before_cleaning = row.original_text,
                q.text = row.clean_text,
                q.cleaning_applied = true,
                q.cleaning_rules_triggered = row.rules,
                q.cleaning_confidence = row.confidence,
                q.needs_review = row.needs_review
            """,
            {"rows": batch}
        )
        written += len(batch)
        logger.info(f"  Written {written:,}/{total:,}")

    logger.info("Cleaning update complete.")


# ─────────────────────────────────────────────
#  Main Execution
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clean metadata and junk from quote text.")
    parser.add_argument("--apply", action="store_true", help="Write changes to DB (default: Dry Run)")
    parser.add_argument("--limit", type=int, default=50000, help="Limit quotes processed (default: 50000)")
    args = parser.parse_args()

    mode = "APPLY MODE" if args.apply else "DRY RUN"
    logger.info("=" * 60)
    logger.info(f"  Track 1 — Quote Text Cleaning Pipeline [{mode}]")
    logger.info("=" * 60)

    client = Neo4jClient(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD, config.NEO4J_DATABASE)

    try:
        # 1. Fetch
        dirty_quotes = fetch_dirty_quotes(client, limit=args.limit)
        
        if not dirty_quotes:
            logger.info("No dirty quotes found matching the patterns. Exiting.")
            return

        # 2. Process
        logger.info("Applying cleaning rules...")
        successes = []
        rule_counts = Counter()
        review_count = 0
        samples_to_print = []

        for q in dirty_quotes:
            clean_text, rules, conf, needs_review = clean_quote(q["text"])
            
            if not rules:
                continue  # No changes made
                
            for r in rules:
                rule_counts[r] += 1
                
            if needs_review:
                review_count += 1

            clean_data = {
                "node_id": q["node_id"],
                "original_text": q["text"],
                "clean_text": clean_text,
                "rules": rules,
                "confidence": conf,
                "needs_review": needs_review
            }
            successes.append(clean_data)

            if len(samples_to_print) < 50:
                samples_to_print.append(clean_data)

        # 3. Report
        logger.info("=" * 60)
        logger.info("CLEANING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Candidate quotes scanned: {len(dirty_quotes):,}")
        logger.info(f"Quotes actually cleaned:  {len(successes):,} ({(len(successes)/max(1, len(dirty_quotes))*100):.1f}%)")
        logger.info(f"Flagged for human review: {review_count:,} (Removed >30% of text or left <10 chars)")

        logger.info("\nRules Triggered:")
        for rule, count in rule_counts.most_common():
            logger.info(f"  {rule:<20} {count:,}")

        if not args.apply:
            logger.info("\nDry Run: Sample Transformations (First 50)")
            logger.info("-" * 60)
            for s in samples_to_print:
                review_tag = "[REVIEW!]" if s["needs_review"] else "[  OK   ]"
                print(f"RAW   : {s['original_text']}")
                print(f"CLEAN : {s['clean_text']}")
                print(f"STATUS: {review_tag} conf={s['confidence']:.2f} rules={','.join(s['rules'])}")
                print("-" * 40)
            logger.info("\nDRY RUN ONLY. Use --apply to write to database.")
        else:
            if successes:
                apply_cleaning(client, successes)
            else:
                logger.info("No successful cleans to apply.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
