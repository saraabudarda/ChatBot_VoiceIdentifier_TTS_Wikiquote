import sys
from pathlib import Path
sys.path.insert(0, str(Path('.')))
from src.database.neo4j_client import Neo4jClient
import config

def main():
    client = Neo4jClient(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD, config.NEO4J_DATABASE)
    try:
        print("="*60)
        print("  AUTHOR EXTRACTION IMPACT REPORT")
        print("="*60)
        
        # 1. Validation Counts
        print("\n[1] Overall Database State")
        res = client.execute_query('''
            MATCH (q:Quote)
            WITH count(q) as total_quotes
            MATCH (q:Quote) WHERE (()-[:SAID]->(q)) WITH total_quotes, count(q) as said_count
            MATCH (q:Quote) WHERE NOT (()-[:SAID]->(q)) WITH total_quotes, said_count, count(q) as orphans
            MATCH (p:Person) WITH total_quotes, said_count, orphans, count(p) as total_persons
            RETURN total_quotes, said_count, orphans, total_persons
        ''')[0]
        print(f"Total Quotes: {res['total_quotes']:,}")
        print(f"Quotes with SAID relationships: {res['said_count']:,}")
        print(f"Remaining Orphan Quotes: {res['orphans']:,}")
        print(f"Total Person nodes: {res['total_persons']:,}")

        # Extraction Stats
        print("\n[2] Extraction Run Impact")
        extracted_quotes = client.execute_query('MATCH (q:Quote) WHERE q.author_extracted = true RETURN count(q) as c')[0]['c']
        needs_review = client.execute_query('MATCH (q:Quote) WHERE q.author_extracted = true AND q.needs_review = true RETURN count(q) as c')[0]['c']
        new_persons = client.execute_query('MATCH (p:Person {created_by_extraction: true}) RETURN count(p) as c')[0]['c']
        print(f"Quotes with author successfully extracted (author_extracted=true): {extracted_quotes:,}")
        print(f"Quotes flagged for review (needs_review=true): {needs_review:,}")
        print(f"New Person nodes created: {new_persons:,}")

        # 3. Bad Extraction Checks
        print("\n[3] Quality Alerts & Bad Extractions (among NEW Person nodes)")
        checks = [
            ("Digits in name", "MATCH (p:Person {created_by_extraction:true}) WHERE p.name =~ '.*\\\\d+.*' RETURN count(p) as c, collect(p.name)[0..5] as s"),
            ("Long names (>30 chars)", "MATCH (p:Person {created_by_extraction:true}) WHERE size(p.name) > 30 RETURN count(p) as c, collect(p.name)[0..5] as s"),
            ("All Caps", "MATCH (p:Person {created_by_extraction:true}) WHERE p.name = toupper(p.name) AND size(p.name)>3 RETURN count(p) as c, collect(p.name)[0..5] as s"),
            ("Single Word", "MATCH (p:Person {created_by_extraction:true}) WHERE NOT p.name CONTAINS ' ' RETURN count(p) as c, collect(p.name)[0..5] as s"),
            ("Junk/Metadata Keywords", "MATCH (p:Person {created_by_extraction:true}) WHERE p.name =~ '(?i).*(read more|unknown|anonymous|page|chapter|vol|http|www|press|book).*' RETURN count(p) as c, collect(p.name)[0..5] as s")
        ]
        for name, query in checks:
            r = client.execute_query(query)[0]
            print(f"- {name}: {r['c']:,} (Samples: {r['s']})")

        # 4. Top 10 Extracted Authors
        print("\n[4] Top 10 New Extracted Authors (by quote count)")
        top_authors = client.execute_query('''
            MATCH (p:Person {created_by_extraction: true})-[:SAID]->(q:Quote)
            RETURN p.name as name, count(q) as c
            ORDER BY c DESC LIMIT 10
        ''')
        for idx, a in enumerate(top_authors):
            print(f"  {idx+1}. {a['name']} ({a['c']})")

        # 5. Samples
        print("\n[5] 50 Before/After Samples")
        samples = client.execute_query('''
            MATCH (p:Person)-[:SAID]->(q:Quote)
            WHERE q.author_extracted = true
            RETURN q.text_before_extraction as original, 
                   q.text as cleaned, 
                   p.name as author, 
                   q.extraction_pattern as pattern, 
                   q.extraction_confidence as conf,
                   coalesce(p.created_by_extraction, false) as is_new
            LIMIT 50
        ''')
        for idx, s in enumerate(samples):
            print(f"--- Sample {idx+1} ---")
            print(f"Original: {s['original']}")
            print(f"Cleaned : {s['cleaned']}")
            print(f"Author  : {s['author']} (New: {s['is_new']})")
            print(f"Meta    : rule={s['pattern']}, conf={s['conf']}")
            
    finally:
        client.close()

if __name__ == '__main__':
    main()
