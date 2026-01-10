"""
Populate database with sample quotes for testing
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.database.neo4j_client import Neo4jClient
from src.database.schema import GraphSchema
import uuid

# Sample quotes data
SAMPLE_QUOTES = [
    {
        'quote': 'To be, or not to be, that is the question.',
        'author': 'William Shakespeare',
        'work': 'Hamlet'
    },
    {
        'quote': 'I think, therefore I am.',
        'author': 'René Descartes',
        'work': 'Discourse on the Method'
    },
    {
        'quote': 'The only thing we have to fear is fear itself.',
        'author': 'Franklin D. Roosevelt',
        'work': 'First Inaugural Address'
    },
    {
        'quote': 'In the middle of difficulty lies opportunity.',
        'author': 'Albert Einstein',
        'work': 'Unknown'
    },
    {
        'quote': 'The unexamined life is not worth living.',
        'author': 'Socrates',
        'work': 'Apology'
    },
    {
        'quote': 'Be yourself; everyone else is already taken.',
        'author': 'Oscar Wilde',
        'work': 'Unknown'
    },
    {
        'quote': 'Two things are infinite: the universe and human stupidity; and I\'m not sure about the universe.',
        'author': 'Albert Einstein',
        'work': 'Unknown'
    },
    {
        'quote': 'A room without books is like a body without a soul.',
        'author': 'Marcus Tullius Cicero',
        'work': 'Unknown'
    },
    {
        'quote': 'You only live once, but if you do it right, once is enough.',
        'author': 'Mae West',
        'work': 'Unknown'
    },
    {
        'quote': 'If you tell the truth, you don\'t have to remember anything.',
        'author': 'Mark Twain',
        'work': 'Unknown'
    }
]

def populate_sample_data():
    """Populate Neo4j with sample quotes."""
    print("Connecting to Neo4j...")
    client = Neo4jClient(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE
    )
    
    print("Creating sample quotes...")
    queries = []
    
    for quote_data in SAMPLE_QUOTES:
        # Generate unique ID
        quote_id = str(uuid.uuid4())
        
        # Create quote node
        quote_query, quote_params = GraphSchema.create_quote_node({
            'id': quote_id,
            'quote_raw': quote_data['quote'],
            'quote_normalized': quote_data['quote'].lower(),
            'language': 'en'
        })
        queries.append((quote_query, quote_params))
        
        # Create author node
        author_query, author_params = GraphSchema.create_author_node(quote_data['author'])
        queries.append((author_query, author_params))
        
        # Create work node if available
        if quote_data['work'] != 'Unknown':
            work_query, work_params = GraphSchema.create_work_node(quote_data['work'])
            queries.append((work_query, work_params))
        
        # Create relationships
        rel_query, rel_params = GraphSchema.create_relationships(
            quote_id, quote_data['author'], quote_data['work']
        )
        queries.append((rel_query, rel_params))
    
    # Execute batch insert
    print(f"Inserting {len(SAMPLE_QUOTES)} sample quotes...")
    client.batch_write(queries, batch_size=100)
    
    # Get statistics
    stats = client.get_statistics()
    print("\n✅ Sample data populated successfully!")
    print(f"\nDatabase Statistics:")
    print(f"  Quotes: {stats['quotes']}")
    print(f"  Authors: {stats['authors']}")
    print(f"  Works: {stats['works']}")
    
    client.close()
    print("\nYou can now launch the Streamlit UI:")
    print("  streamlit run src/ui/streamlit_app.py")

if __name__ == '__main__':
    populate_sample_data()
