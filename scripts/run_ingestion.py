"""
Enhanced Data Ingestion Pipeline with Role Taxonomy Support

Processes Wikiquote XML dump using enhanced parser with entity type detection.
"""
import logging
import argparse
from pathlib import Path
import uuid
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.ingestion.xml_parser import EnhancedWikiquoteParser
from src.ingestion.text_cleaner import TextCleaner
from src.ingestion.nlp_processor import NLPProcessor
from src.database.neo4j_client import Neo4jClient
from src.database.schema import EnhancedGraphSchema

logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def run_ingestion(limit: int = None, batch_size: int = 1000, clear_db: bool = False):
    """
    Run the complete data ingestion pipeline.
    
    Args:
        limit: Maximum number of pages to process
        batch_size: Number of records to process in each batch
        clear_db: Whether to clear the database before ingestion
    """
    logger.info("=" * 60)
    logger.info("WIKIQUOTE DATA INGESTION PIPELINE (ENHANCED)")
    logger.info("=" * 60)
    
    # Initialize components
    logger.info("Initializing components...")
    parser = EnhancedWikiquoteParser(config.XML_FILE)
    cleaner = TextCleaner(
        language_filter=config.LANGUAGE_FILTER,
        dedup_threshold=config.DEDUP_THRESHOLD
    )
    nlp_processor = NLPProcessor(model_name=config.SPACY_MODEL)
    
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
        client.execute_query(query)
    
    for query in EnhancedGraphSchema.get_index_queries():
        client.execute_query(query)
    
    # Create full-text index
    try:
        client.execute_query(EnhancedGraphSchema.get_fulltext_index_query())
    except Exception as e:
        logger.warning(f"Full-text index creation warning: {e}")
    
    # Process data
    logger.info("Starting data processing...")
    
    processed_count = 0
    inserted_count = 0
    batch = []
    
    for quote_data in parser.parse(limit=limit):
        # Clean text
        cleaned_text = cleaner.clean(quote_data['quote_raw'])
        if not cleaned_text:
            continue
        
        quote_data['quote_raw'] = cleaned_text
        
        # NLP processing
        processed_data = nlp_processor.process(quote_data)
        
        # Prepare for database insertion
        entity_type = processed_data.get('entity_type', 'Source')
        
        if entity_type == 'Person':
            query, params = EnhancedGraphSchema.create_person_with_quote(processed_data)
        elif entity_type == 'Work':
            query, params = EnhancedGraphSchema.create_work_with_quote(processed_data)
        else:
            query, params = EnhancedGraphSchema.create_source_with_quote(processed_data)
        
        batch.append((query, params))
        processed_count += 1
        
        # Batch insert
        if len(batch) >= batch_size:
            client.batch_write(batch)
            inserted_count += len(batch)
            logger.info(f"Progress: {processed_count} processed, {inserted_count} inserted")
            batch = []
    
    # Insert remaining batch
    if batch:
        client.batch_write(batch)
        inserted_count += len(batch)
    
    # Final statistics
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total quotes processed: {processed_count}")
    logger.info(f"Total quotes inserted: {inserted_count}")
    
    stats = client.get_statistics()
    logger.info("\nDatabase Statistics:")
    logger.info(f"  quotes: {stats['quotes']}")
    logger.info(f"  authors: {stats['authors']}")
    logger.info(f"  works: {stats['works']}")
    
    client.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Ingest Wikiquote data into Neo4j')
    parser.add_argument('--limit', type=int, help='Limit number of pages to process')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing')
    parser.add_argument('--clear', action='store_true', help='Clear database before ingestion')
    
    args = parser.parse_args()
    
    run_ingestion(
        limit=args.limit,
        batch_size=args.batch_size,
        clear_db=args.clear
    )


if __name__ == '__main__':
    main()
