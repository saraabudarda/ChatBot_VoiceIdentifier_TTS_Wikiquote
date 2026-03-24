import sys
from pathlib import Path
sys.path.insert(0, str(Path('.')))
from src.database.neo4j_client import Neo4jClient
import config
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def main():
    client = Neo4jClient(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD, config.NEO4J_DATABASE)
    try:
        logger.info("="*60)
        logger.info("  ROLLING BACK AUTHOR EXTRACTION")
        logger.info("="*60)
        
        # 1. Delete relationships pointing to newly extracted quotes (they were orphans before)
        logger.info("1. Removing SAID relationships from extracted quotes...")
        res_rel = client.execute_query('''
            MATCH ()-[r:SAID]->(q:Quote)
            WHERE q.author_extracted = true
            DELETE r
            RETURN count(r) as c
        ''')
        rels_deleted = res_rel[0]['c'] if res_rel else 0
        logger.info(f"   Removed {rels_deleted:,} SAID relationships.")

        # 2. Delete newly created Person nodes
        logger.info("2. Deleting Person nodes created by extraction...")
        res_person = client.execute_query('''
            MATCH (p:Person {created_by_extraction: true})
            DETACH DELETE p
            RETURN count(p) as c
        ''')
        persons_deleted = res_person[0]['c'] if res_person else 0
        logger.info(f"   Deleted {persons_deleted:,} Person nodes.")

        # 3. Restore Quote text and remove extraction properties
        logger.info("3. Restoring Quote text and removing extraction metadata...")
        last_id = -1
        quotes_restored = 0
        while True:
            batch = client.execute_query('''
                MATCH (q:Quote)
                WHERE id(q) > $last_id AND q.author_extracted = true
                WITH q ORDER BY id(q) LIMIT 10000
                WITH collect(q) as quotes, max(id(q)) as max_id
                UNWIND quotes as q
                SET q.text = coalesce(q.text_before_extraction, q.text)
                REMOVE q.author_extracted, 
                       q.extracted_author_name, 
                       q.extraction_pattern, 
                       q.extraction_confidence, 
                       q.text_before_extraction,
                       q.created_by_extraction
                RETURN max_id, count(q) as c
            ''', {'last_id': last_id})
            
            if not batch or batch[0]['c'] == 0:
                break
                
            quotes_restored += batch[0]['c']
            last_id = batch[0]['max_id']
            logger.info(f"   Restored {quotes_restored:,} quotes...")
            
        logger.info(f"   Total Quotes restored: {quotes_restored:,}")

        # 4. Validation
        logger.info("\n--- VALIDATION RUN ---")
        orphans = client.execute_query('''
            MATCH (q:Quote) WHERE NOT (()-[:SAID]->(q)) RETURN count(q) as c
        ''')[0]['c']
        logger.info(f"Remaining orphan quotes: {orphans:,}")
        
        extracted_persons = client.execute_query('''
            MATCH (p:Person {created_by_extraction: true}) RETURN count(p) as c
        ''')[0]['c']
        logger.info(f"Person nodes with created_by_extraction=true: {extracted_persons:,}")
        
        extraction_flags = client.execute_query('''
            MATCH (q:Quote) WHERE q.author_extracted = true RETURN count(q) as c
        ''')[0]['c']
        logger.info(f"Quotes with author_extracted=true: {extraction_flags:,}")

        logger.info("\n--- SAMPLE 20 QUOTES RESTORED ---")
        samples = client.execute_query('''
            MATCH (q:Quote)
            WHERE NOT (()-[:SAID]->(q))
            RETURN q.text as text
            LIMIT 20
        ''')
        for idx, s in enumerate(samples):
            # Replacing newlines to display nicely in a single line
            disp_text = s['text'].replace('|', ' ').replace('\\n', ' ')
            logger.info(f"  {idx+1}. {disp_text[:100]}")

        logger.info("\nRollback complete and safe.")

    finally:
        client.close()

if __name__ == '__main__':
    main()
