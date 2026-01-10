"""
FAST Ingestion Pipeline - Optimized for Speed

Skips NLP processing and heavy text cleaning to maximize ingestion speed.
Goal: Populate 1.5M quotes quickly.
"""
import logging
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.ingestion.xml_parser import EnhancedWikiquoteParser
from src.database.neo4j_client import Neo4jClient
from src.database.schema import EnhancedGraphSchema

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_fast_ingestion(limit: int = None, batch_size: int = 5000, clear_db: bool = False):
    """
    Fast ingestion pipeline - minimal processing for maximum speed.
    
    Args:
        limit: Maximum number of pages to process
        batch_size: Number of records per batch (larger = faster)
        clear_db: Whether to clear database first
    """
    logger.info("=" * 70)
    logger.info("FAST WIKIQUOTE INGESTION - OPTIMIZED FOR SPEED")
    logger.info("=" * 70)
    
    # Initialize parser only
    logger.info("Initializing parser...")
    parser = EnhancedWikiquoteParser(config.XML_FILE)
    
    # Connect to Neo4j
    logger.info("Connecting to Neo4j...")
    client = Neo4jClient(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE
    )
    
    # Clear database if requested
    if clear_db:
        logger.info("Clearing database...")
        client.clear_database()
    
    # Create schema
    logger.info("Creating database schema...")
    for query in EnhancedGraphSchema.get_constraint_queries():
        try:
            client.execute_query(query)
        except Exception as e:
            logger.debug(f"Constraint creation: {e}")
    
    for query in EnhancedGraphSchema.get_index_queries():
        try:
            client.execute_query(query)
        except Exception as e:
            logger.debug(f"Index creation: {e}")
    
    # Create full-text index
    try:
        client.execute_query(EnhancedGraphSchema.get_fulltext_index_query())
    except Exception as e:
        logger.debug(f"Full-text index: {e}")
    
    # Process data - FAST MODE
    logger.info("Starting FAST data processing...")
    logger.info("Skipping: NLP processing, heavy text cleaning, deduplication")
    logger.info(f"Batch size: {batch_size} (larger batches = faster)")
    
    processed_count = 0
    inserted_count = 0
    batch = []
    seen_quotes = set()  # Simple exact duplicate check only
    
    for quote_data in parser.parse(limit=limit):
        # Minimal cleaning - just strip whitespace
        quote_text = quote_data['quote_raw'].strip()
        
        # Skip very short quotes
        if len(quote_text) < 15:
            continue
        
        # Simple exact duplicate check (fast)
        if quote_text.lower() in seen_quotes:
            continue
        seen_quotes.add(quote_text.lower())
        
        # Prepare for database - NO NLP PROCESSING
        quote_data['quote_raw'] = quote_text
        quote_data['quote_normalized'] = quote_text.lower()  # Simple lowercase
        
        entity_type = quote_data.get('entity_type', 'Source')
        
        if entity_type == 'Person':
            query, params = EnhancedGraphSchema.create_person_with_quote(quote_data)
        elif entity_type == 'Work':
            query, params = EnhancedGraphSchema.create_work_with_quote(quote_data)
        else:
            query, params = EnhancedGraphSchema.create_source_with_quote(quote_data)
        
        batch.append((query, params))
        processed_count += 1
        
        # Batch insert with larger batches
        if len(batch) >= batch_size:
            try:
                client.batch_write(batch, batch_size=batch_size)
                inserted_count += len(batch)
                logger.info(f"Progress: {processed_count:,} processed, {inserted_count:,} inserted")
                batch = []
            except Exception as e:
                logger.error(f"Batch write error: {e}")
                batch = []
    
    # Insert remaining batch
    if batch:
        try:
            client.batch_write(batch, batch_size=batch_size)
            inserted_count += len(batch)
        except Exception as e:
            logger.error(f"Final batch error: {e}")
    
    # Final statistics
    logger.info("=" * 70)
    logger.info("FAST INGESTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total quotes processed: {processed_count:,}")
    logger.info(f"Total quotes inserted: {inserted_count:,}")
    
    stats = client.get_statistics()
    logger.info("\nDatabase Statistics:")
    logger.info(f"  Quotes: {stats['quotes']:,}")
    logger.info(f"  Authors: {stats['authors']:,}")
    logger.info(f"  Works: {stats['works']:,}")
    
    client.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Fast Wikiquote ingestion')
    parser.add_argument('--limit', type=int, help='Limit number of pages')
    parser.add_argument('--batch-size', type=int, default=5000, help='Batch size (default: 5000)')
    parser.add_argument('--clear', action='store_true', help='Clear database first')
    
    args = parser.parse_args()
    
    run_fast_ingestion(
        limit=args.limit,
        batch_size=args.batch_size,
        clear_db=args.clear
    )


if __name__ == '__main__':
    main()
