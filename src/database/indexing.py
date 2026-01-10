"""
Full-Text Index Management for Neo4j

This module manages the creation and verification of full-text
search indexes for quote retrieval and autocompletion.
"""
import logging
from typing import List, Dict, Optional
from .neo4j_client import Neo4jClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndexManager:
    """
    Manages full-text search indexes for the Wikiquote database.
    
    Handles creation, verification, and querying of full-text indexes
    for efficient quote search and autocompletion.
    """
    
    def __init__(self, client: Neo4jClient, index_name: str = 'quoteTextIndex'):
        """
        Initialize the index manager.
        
        Args:
            client: Neo4jClient instance
            index_name: Name of the full-text index
        """
        self.client = client
        self.index_name = index_name
    
    def create_fulltext_index(self):
        """Create the full-text search index for quotes."""
        query = f"""
        CREATE FULLTEXT INDEX {self.index_name} IF NOT EXISTS
        FOR (q:Quote)
        ON EACH [q.text]
        """
        
        try:
            self.client.execute_query(query)
            logger.info(f"Created full-text index: {self.index_name}")
        except Exception as e:
            logger.warning(f"Full-text index creation failed (may already exist): {e}")
    
    def verify_index(self) -> bool:
        """
        Verify that the full-text index exists.
        
        Returns:
            True if index exists, False otherwise
        """
        query = "SHOW INDEXES"
        
        try:
            result = self.client.execute_query(query)
            
            for record in result:
                if record.get('name') == self.index_name:
                    logger.info(f"Full-text index '{self.index_name}' exists")
                    return True
            
            logger.warning(f"Full-text index '{self.index_name}' not found")
            return False
        except Exception as e:
            logger.error(f"Failed to verify index: {e}")
            return False
    
    def search_quotes(self, query_text: str, limit: int = 10) -> List[Dict]:
        """
        Search quotes using the full-text index.
        
        Args:
            query_text: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching quotes with metadata
        """
        query = f"""
        CALL db.index.fulltext.queryNodes($index_name, $query_text)
        YIELD node, score
        MATCH (p)-[:HAS_QUOTE|SAID]->(node)
        RETURN node.text AS quote,
               p.name AS author,
               score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        params = {
            'index_name': self.index_name,
            'query_text': query_text,
            'limit': limit
        }
        
        try:
            results = self.client.execute_query(query, params)
            logger.info(f"Found {len(results)} results for query: '{query_text}'")
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def fuzzy_search(self, query_text: str, limit: int = 10) -> List[Dict]:
        """
        Perform fuzzy search on quotes.
        
        Args:
            query_text: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching quotes with metadata
        """
        # Add fuzzy operator (~) to each term
        fuzzy_query = ' '.join([f"{term}~" for term in query_text.split()])
        
        return self.search_quotes(fuzzy_query, limit)
    
    def autocomplete(self, partial_quote: str, limit: int = 5) -> List[Dict]:
        """
        Autocomplete a partial quote.
        
        Args:
            partial_quote: Partial quote text
            limit: Maximum number of suggestions
            
        Returns:
            List of quote suggestions
        """
        # Use wildcard for prefix matching
        query_text = f"{partial_quote}*"
        
        return self.search_quotes(query_text, limit)
    
    def drop_index(self):
        """Drop the full-text index."""
        query = f"DROP INDEX {self.index_name} IF EXISTS"
        
        try:
            self.client.execute_query(query)
            logger.info(f"Dropped full-text index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to drop index: {e}")


def main():
    """Test the index manager."""
    import config
    from .neo4j_client import Neo4jClient
    
    client = Neo4jClient(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE
    )
    
    manager = IndexManager(client, config.FULLTEXT_INDEX_NAME)
    
    print("Testing IndexManager...")
    
    # Verify index
    if manager.verify_index():
        print("✓ Full-text index exists")
        
        # Test search
        results = manager.search_quotes("to be or not to be", limit=3)
        print(f"\nSearch results ({len(results)}):")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['quote'][:80]}...")
            print(f"   Author: {result['author']}, Score: {result['score']:.2f}")
    else:
        print("✗ Full-text index not found")
        print("Creating index...")
        manager.create_fulltext_index()
    
    client.close()


if __name__ == '__main__':
    main()
