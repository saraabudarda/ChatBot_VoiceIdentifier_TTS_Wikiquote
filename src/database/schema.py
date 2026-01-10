"""
Enhanced Neo4j Graph Schema with Person/Work/Source Entity Types

Supports role taxonomy and improved entity classification.
"""
from typing import List, Dict


class EnhancedGraphSchema:
    """
    Enhanced graph schema supporting Person, Work, and Source entities.
    
    Nodes: Quote, Person, Work, Source
    Relationships: SAID, HAS_QUOTE, WROTE, FROM_WORK
    """
    
    # Node labels
    QUOTE = 'Quote'
    PERSON = 'Person'
    WORK = 'Work'
    SOURCE = 'Source'
    AUTHOR = 'Author'  # Legacy compatibility
    
    # Relationship types
    SAID = 'SAID'
    HAS_QUOTE = 'HAS_QUOTE'
    WROTE = 'WROTE'
    FROM_WORK = 'FROM_WORK'
    
    @staticmethod
    def get_constraint_queries() -> List[str]:
        """Get Cypher queries to create constraints."""
        return [
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (p:{EnhancedGraphSchema.PERSON}) REQUIRE p.name IS UNIQUE",
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (w:{EnhancedGraphSchema.WORK}) REQUIRE w.name IS UNIQUE",
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (s:{EnhancedGraphSchema.SOURCE}) REQUIRE s.name IS UNIQUE",
        ]
    
    @staticmethod
    def get_index_queries() -> List[str]:
        """Get Cypher queries to create indexes."""
        return [
            f"CREATE INDEX IF NOT EXISTS FOR (q:{EnhancedGraphSchema.QUOTE}) ON (q.text)",
            f"CREATE INDEX IF NOT EXISTS FOR (p:{EnhancedGraphSchema.PERSON}) ON (p.name)",
            f"CREATE INDEX IF NOT EXISTS FOR (w:{EnhancedGraphSchema.WORK}) ON (w.name)",
        ]
    
    @staticmethod
    def get_fulltext_index_query() -> str:
        """Get Cypher query to create full-text search index."""
        return """
        CREATE FULLTEXT INDEX quote_search IF NOT EXISTS
        FOR (q:Quote) ON EACH [q.text]
        """
    
    @staticmethod
    def create_person_with_quote(quote_data: Dict) -> tuple:
        """
        Create Person node with quote and relationships.
        
        Args:
            quote_data: Dictionary with quote and person information
            
        Returns:
            Tuple of (query, parameters)
        """
        query = """
        MERGE (p:Person {name: $author})
        SET p.roles = $roles
        MERGE (q:Quote {text: $quote})
        MERGE (p)-[:HAS_QUOTE]->(q)
        MERGE (p)-[:SAID]->(q)
        RETURN p, q
        """
        
        params = {
            'author': quote_data.get('author'),
            'roles': quote_data.get('roles', []),
            'quote': quote_data.get('quote_raw')
        }
        
        return query, params
    
    @staticmethod
    def create_work_with_quote(quote_data: Dict) -> tuple:
        """
        Create Work node with quote and link to real author.
        
        Args:
            quote_data: Dictionary with quote and work information
            
        Returns:
            Tuple of (query, parameters)
        """
        query = """
        MERGE (w:Work {name: $work_name})
        
        FOREACH (_ IN CASE WHEN $real_author IS NOT NULL THEN [1] ELSE [] END |
            MERGE (creator:Person {name: $real_author})
            MERGE (creator)-[:WROTE]->(w)
        )
        
        MERGE (q:Quote {text: $quote})
        MERGE (w)-[:HAS_QUOTE]->(q)
        RETURN w, q
        """
        
        params = {
            'work_name': quote_data.get('author'),
            'real_author': quote_data.get('real_author'),
            'quote': quote_data.get('quote_raw')
        }
        
        return query, params
    
    @staticmethod
    def create_source_with_quote(quote_data: Dict) -> tuple:
        """
        Create generic Source node with quote.
        
        Args:
            quote_data: Dictionary with quote and source information
            
        Returns:
            Tuple of (query, parameters)
        """
        query = """
        MERGE (s:Source {name: $source_name})
        SET s.type = $entity_type
        MERGE (q:Quote {text: $quote})
        MERGE (s)-[:HAS_QUOTE]->(q)
        RETURN s, q
        """
        
        params = {
            'source_name': quote_data.get('author'),
            'entity_type': quote_data.get('entity_type', 'Source'),
            'quote': quote_data.get('quote_raw')
        }
        
        return query, params


# Maintain backward compatibility
GraphSchema = EnhancedGraphSchema
