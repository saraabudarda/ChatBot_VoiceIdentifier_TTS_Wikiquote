"""
Neo4j Client for Database Operations

This module provides connection management and CRUD operations
for the Neo4j graph database.
"""
import logging
from typing import List, Dict, Optional, Any
from neo4j import GraphDatabase, Session, Transaction
from neo4j.exceptions import ServiceUnavailable, AuthError
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Client for Neo4j database operations.
    
    Handles connection management, transactions, batch operations,
    and error handling with retry logic.
    """
    
    def __init__(self, uri: str, user: str, password: str, database: str = 'neo4j'):
        """
        Initialize Neo4j client.
        
        Args:
            uri: Neo4j connection URI (e.g., neo4j://localhost:7687)
            user: Username for authentication
            password: Password for authentication
            database: Database name (default: neo4j)
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self._connect()
    
    def _connect(self, max_retries: int = 3):
        """
        Establish connection to Neo4j with retry logic.
        
        Args:
            max_retries: Maximum number of connection attempts
        """
        for attempt in range(max_retries):
            try:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password)
                )
                # Test connection
                self.driver.verify_connectivity()
                logger.info(f"Connected to Neo4j at {self.uri}")
                return
            except (ServiceUnavailable, AuthError) as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise Exception(f"Failed to connect to Neo4j after {max_retries} attempts")
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def test_connection(self) -> bool:
        """
        Test if connection is active.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """
        Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        parameters = parameters or {}
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters)
            return [dict(record) for record in result]
    
    def execute_write(self, query: str, parameters: Dict = None) -> Any:
        """
        Execute a write query in a transaction.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            Query result
        """
        parameters = parameters or {}
        
        with self.driver.session(database=self.database) as session:
            return session.execute_write(
                lambda tx: tx.run(query, parameters).single()
            )
    
    def batch_write(self, queries: List[tuple], batch_size: int = 1000):
        """
        Execute multiple write queries in batches.
        
        Args:
            queries: List of (query, parameters) tuples
            batch_size: Number of queries per transaction
        """
        total = len(queries)
        processed = 0
        
        with self.driver.session(database=self.database) as session:
            for i in range(0, total, batch_size):
                batch = queries[i:i + batch_size]
                
                def execute_batch(tx: Transaction):
                    for query, params in batch:
                        tx.run(query, params)
                
                session.execute_write(execute_batch)
                processed += len(batch)
                
                if processed % 5000 == 0:
                    logger.info(f"Processed {processed}/{total} queries")
        
        logger.info(f"Batch write complete: {processed} queries executed")
    
    def create_constraints(self, constraint_queries: List[str]):
        """
        Create database constraints.
        
        Args:
            constraint_queries: List of constraint creation queries
        """
        for query in constraint_queries:
            try:
                self.execute_query(query)
                logger.info(f"Created constraint: {query[:50]}...")
            except Exception as e:
                logger.warning(f"Constraint creation failed (may already exist): {e}")
    
    def create_indexes(self, index_queries: List[str]):
        """
        Create database indexes.
        
        Args:
            index_queries: List of index creation queries
        """
        for query in index_queries:
            try:
                self.execute_query(query)
                logger.info(f"Created index: {query[:50]}...")
            except Exception as e:
                logger.warning(f"Index creation failed (may already exist): {e}")
    
    def count_nodes(self, label: str) -> int:
        """
        Count nodes with a specific label.
        
        Args:
            label: Node label
            
        Returns:
            Number of nodes
        """
        query = f"MATCH (n:{label}) RETURN count(n) AS count"
        result = self.execute_query(query)
        return result[0]['count'] if result else 0
    
    def count_relationships(self, rel_type: str) -> int:
        """
        Count relationships of a specific type.
        
        Args:
            rel_type: Relationship type
            
        Returns:
            Number of relationships
        """
        query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count"
        result = self.execute_query(query)
        return result[0]['count'] if result else 0
    
    def clear_database(self):
        """
        Clear all nodes and relationships from the database.
        
        WARNING: This deletes all data!
        """
        logger.warning("Clearing database - all data will be deleted!")
        query = "MATCH (n) DETACH DELETE n"
        self.execute_query(query)
        logger.info("Database cleared")
    
    def get_statistics(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with node and relationship counts
        """
        stats = {
            'quotes': self.count_nodes('Quote'),
            'authors': self.count_nodes('Author'),
            'works': self.count_nodes('Work'),
            'categories': self.count_nodes('Category'),
            'said_relationships': self.count_relationships('SAID'),
            'from_work_relationships': self.count_relationships('FROM_WORK'),
        }
        return stats


def main():
    """Test the Neo4j client."""
    import config
    
    client = Neo4jClient(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE
    )
    
    print("Testing Neo4j connection...")
    if client.test_connection():
        print("✓ Connection successful")
        
        # Get statistics
        stats = client.get_statistics()
        print("\nDatabase Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("✗ Connection failed")
    
    client.close()


if __name__ == '__main__':
    main()
