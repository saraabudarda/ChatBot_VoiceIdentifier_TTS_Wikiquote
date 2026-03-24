"""
Data Quality Cleaning Pipeline

Processes existing quotes to create a high-quality dataset:
- Removes duplicates (exact and near-duplicates)
- Filters out low-quality quotes (too short, metadata, URLs)
- Validates quote structure and content
- Improves attributions (links works to authors)
- Target: ~950K high-quality quotes from 1.3M
"""
import logging
import re
from typing import List, Dict, Set
from difflib import SequenceMatcher
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.database.neo4j_client import Neo4jClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QualityFilter:
    """Filters and validates quote quality."""
    
    def __init__(self, min_length: int = 20, max_length: int = 500):
        self.min_length = min_length
        self.max_length = max_length
        self.seen_quotes = set()
        self.seen_normalized = {}
    
    def is_high_quality(self, quote: str) -> bool:
        """
        Check if quote meets quality standards.
        
        Args:
            quote: Quote text
            
        Returns:
            True if high quality, False otherwise
        """
        # Length check
        if len(quote) < self.min_length or len(quote) > self.max_length:
            return False
        
        # Remove quotes that are mostly URLs
        if 'http://' in quote or 'https://' in quote or 'www.' in quote:
            return False
        
        # Remove metadata patterns
        metadata_patterns = [
            r'^\[.*\]$',  # [Category: something]
            r'^File:',
            r'^Image:',
            r'^Retrieved',
            r'^Archived',
            r'^\{\{',  # Template markup
            r'^\|',  # Table markup
        ]
        
        for pattern in metadata_patterns:
            if re.match(pattern, quote, re.IGNORECASE):
                return False
        
        # Must have at least 50% alphabetic characters
        alpha_chars = sum(c.isalpha() for c in quote)
        if alpha_chars < len(quote) * 0.5:
            return False
        
        # Must have at least 3 words
        words = quote.split()
        if len(words) < 3:
            return False
        
        # Check for excessive special characters
        special_chars = sum(1 for c in quote if c in '{}[]|<>')
        if special_chars > 5:
            return False
        
        return True
    
    def is_duplicate(self, quote: str, threshold: float = 0.95) -> bool:
        """
        Check if quote is a duplicate or near-duplicate.
        
        Args:
            quote: Quote text
            threshold: Similarity threshold (0.95 = 95% similar)
            
        Returns:
            True if duplicate, False otherwise
        """
        # Exact duplicate check
        quote_lower = quote.lower().strip()
        if quote_lower in self.seen_quotes:
            return True
        
        # Near-duplicate check (expensive, so limit to recent quotes)
        for seen_quote in list(self.seen_normalized.keys())[-1000:]:
            similarity = SequenceMatcher(None, quote_lower, seen_quote).ratio()
            if similarity >= threshold:
                return True
        
        # Not a duplicate - add to seen set
        self.seen_quotes.add(quote_lower)
        self.seen_normalized[quote_lower] = True
        
        return False


def clean_database(client: Neo4jClient, batch_size: int = 1000):
    """
    Clean the database by removing low-quality and duplicate quotes.
    
    Args:
        client: Neo4j client
        batch_size: Number of quotes to process per batch
    """
    logger.info("=" * 70)
    logger.info("DATA QUALITY CLEANING PIPELINE")
    logger.info("=" * 70)
    
    # Get initial count
    initial_stats = client.get_statistics()
    initial_count = initial_stats['quotes']
    logger.info(f"Initial quote count: {initial_count:,}")
    
    # Initialize quality filter
    quality_filter = QualityFilter(min_length=20, max_length=500)
    
    # Process quotes in batches
    offset = 0
    processed = 0
    deleted = 0
    
    while True:
        # Fetch batch of quotes
        query = """
        MATCH (q:Quote)
        RETURN id(q) AS node_id, q.text AS text
        SKIP $offset
        LIMIT $limit
        """
        
        results = client.execute_query(query, {'offset': offset, 'limit': batch_size})
        
        if not results:
            break
        
        # Identify quotes to delete
        to_delete = []
        
        for record in results:
            quote_text = record['text']
            node_id = record['node_id']
            processed += 1
            
            # Check quality
            if not quality_filter.is_high_quality(quote_text):
                to_delete.append(node_id)
                continue
            
            # Check for duplicates
            if quality_filter.is_duplicate(quote_text):
                to_delete.append(node_id)
        
        # Delete low-quality quotes
        if to_delete:
            delete_query = """
            MATCH (q:Quote)
            WHERE id(q) IN $node_ids
            DETACH DELETE q
            """
            client.execute_query(delete_query, {'node_ids': to_delete})
            deleted += len(to_delete)
        
        logger.info(f"Processed: {processed:,} | Deleted: {deleted:,} | Remaining: ~{processed - deleted:,}")
        
        offset += batch_size
    
    # Get final count
    final_stats = client.get_statistics()
    final_count = final_stats['quotes']
    
    logger.info("=" * 70)
    logger.info("CLEANING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Initial quotes: {initial_count:,}")
    logger.info(f"Final quotes: {final_count:,}")
    logger.info(f"Removed: {initial_count - final_count:,}")
    logger.info(f"Quality improvement: {(initial_count - final_count) / initial_count * 100:.1f}% reduction")


def main():
    """Main entry point."""
    logger.info("Connecting to Neo4j...")
    client = Neo4jClient(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE
    )
    
    try:
        clean_database(client, batch_size=5000)
    finally:
        client.close()


if __name__ == '__main__':
    main()
