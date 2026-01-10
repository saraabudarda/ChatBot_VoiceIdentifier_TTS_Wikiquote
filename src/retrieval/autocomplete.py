"""
Quote Autocompletion Engine

This module implements quote retrieval and autocompletion using
Neo4j full-text search with ranking and relevance scoring.
"""
import logging
from typing import List, Dict, Optional
from ..database.neo4j_client import Neo4jClient
from ..database.indexing import IndexManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuoteAutocomplete:
    """
    Quote autocompletion and retrieval engine.
    
    Provides methods for completing partial quotes, finding similar quotes,
    and retrieving quote metadata including author and source work.
    """
    
    def __init__(self, client: Neo4jClient, index_name: str = 'quoteTextIndex'):
        """
        Initialize the autocomplete engine.
        
        Args:
            client: Neo4jClient instance
            index_name: Name of the full-text index
        """
        self.client = client
        self.index_manager = IndexManager(client, 'quote_search')
    
    def complete_quote(self, partial_quote: str, max_results: int = 10) -> List[Dict]:
        """
        Complete a partial quote.
        
        Args:
            partial_quote: Partial quote text
            max_results: Maximum number of results to return
            
        Returns:
            List of matching quotes with metadata
        """
        if not partial_quote or len(partial_quote) < 3:
            return []
        
        # Search using full-text index
        results = self.index_manager.search_quotes(partial_quote, limit=max_results)
        
        # Enhance results with ranking
        ranked_results = self._rank_results(partial_quote, results)
        
        return ranked_results
    
    def find_by_author(self, author_name: str, limit: int = 10) -> List[Dict]:
        """
        Find quotes by a specific author.
        
        Args:
            author_name: Name of the author
            limit: Maximum number of results
            
        Returns:
            List of quotes by the author
        """
        query = """
        MATCH (p:Person)-[:HAS_QUOTE|SAID]->(q:Quote)
        WHERE toLower(p.name) CONTAINS toLower($author_name)
        RETURN q.text AS quote,
               p.name AS author
        LIMIT $limit
        """
        
        params = {'author_name': author_name, 'limit': limit}
        results = self.client.execute_query(query, params)
        
        return results
    
    def find_by_work(self, work_title: str, limit: int = 10) -> List[Dict]:
        """
        Find quotes from a specific work.
        
        Args:
            work_title: Title of the work
            limit: Maximum number of results
            
        Returns:
            List of quotes from the work
        """
        query = """
        MATCH (w:Work)-[:HAS_QUOTE]->(q:Quote)
        WHERE toLower(w.name) CONTAINS toLower($work_title)
        RETURN q.text AS quote,
               w.name AS author
        LIMIT $limit
        """
        
        params = {'work_title': work_title, 'limit': limit}
        results = self.client.execute_query(query, params)
        
        return results
    
    def get_random_quotes(self, count: int = 5) -> List[Dict]:
        """
        Get random quotes from the database.
        
        Args:
            count: Number of random quotes to retrieve
            
        Returns:
            List of random quotes
        """
        query = """
        MATCH (p)-[:HAS_QUOTE|SAID]->(q:Quote)
        WITH q, p, rand() AS random
        ORDER BY random
        LIMIT $count
        RETURN q.text AS quote,
               p.name AS author
        """
        
        params = {'count': count}
        results = self.client.execute_query(query, params)
        
        return results
    
    def fuzzy_search(self, query_text: str, max_results: int = 10) -> List[Dict]:
        """
        Perform fuzzy search for quotes.
        
        Args:
            query_text: Search query
            max_results: Maximum number of results
            
        Returns:
            List of matching quotes
        """
        results = self.index_manager.fuzzy_search(query_text, limit=max_results)
        return self._rank_results(query_text, results)
    
    def _rank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Rank search results by relevance with quality boosting.
        
        Ranking factors:
        - Base full-text search score
        - Prefer shorter, memorable quotes
        - Boost quotes with known authors
        - Boost quotes linked to works
        - Penalize very long quotes
        - Prefix match bonus
        
        Args:
            query: Original search query
            results: List of search results
            
        Returns:
            Ranked list of results
        """
        query_lower = query.lower()
        query_len = len(query)
        
        for result in results:
            quote = result.get('quote', '')
            quote_lower = quote.lower()
            author = result.get('author', 'Unknown')
            work = result.get('work', None)
            
            # Base score from full-text search
            base_score = result.get('score', 0)
            
            # 1. Length preference (prefer shorter, memorable quotes)
            quote_len = len(quote)
            if quote_len < 100:
                # Short quotes are often more memorable
                length_score = 1.5
            elif quote_len < 200:
                # Medium quotes are good
                length_score = 1.2
            elif quote_len < 300:
                # Longer quotes are okay
                length_score = 1.0
            else:
                # Very long quotes are penalized
                length_score = 0.7
            
            # 2. Author quality boost
            if author and author != 'Unknown' and len(author) > 2:
                # Known author - boost significantly
                author_boost = 1.3
            else:
                # Unknown or generic author
                author_boost = 0.8
            
            # 3. Work attribution boost
            if work and work != 'Unknown':
                # Quote has source work - boost
                work_boost = 1.2
            else:
                # No work attribution
                work_boost = 1.0
            
            # 4. Prefix match bonus (exact start match)
            if quote_lower.startswith(query_lower):
                prefix_bonus = 1.5
            elif query_lower in quote_lower[:50]:
                # Query appears in first 50 chars
                prefix_bonus = 1.3
            else:
                prefix_bonus = 1.0
            
            # 5. Query coverage (how much of quote matches query)
            query_words = set(query_lower.split())
            quote_words = set(quote_lower.split())
            if query_words and quote_words:
                coverage = len(query_words & quote_words) / len(query_words)
                coverage_score = 1.0 + (coverage * 0.5)
            else:
                coverage_score = 1.0
            
            # Calculate final score with weighted factors
            final_score = (
                base_score * 0.4 +          # Full-text relevance (40%)
                length_score * 0.2 +         # Length preference (20%)
                author_boost * 0.15 +        # Author quality (15%)
                work_boost * 0.1 +           # Work attribution (10%)
                coverage_score * 0.15        # Query coverage (15%)
            ) * prefix_bonus                 # Prefix match multiplier
            
            result['final_score'] = final_score
            result['_debug'] = {
                'base': base_score,
                'length': length_score,
                'author': author_boost,
                'work': work_boost,
                'prefix': prefix_bonus,
                'coverage': coverage_score
            }
        
        # Sort by final score
        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        return results


def main():
    """Test the autocomplete engine."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    import config
    from src.database.neo4j_client import Neo4jClient
    
    client = Neo4jClient(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE
    )
    
    autocomplete = QuoteAutocomplete(client, config.FULLTEXT_INDEX_NAME)
    
    print("Testing Quote Autocomplete:\n")
    
    # Test completion
    test_query = "to be or not"
    print(f"Query: '{test_query}'")
    results = autocomplete.complete_quote(test_query, max_results=3)
    
    print(f"\nResults ({len(results)}):")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['quote']}")
        print(f"   Author: {result['author']}")
        print(f"   Work: {result.get('work', 'Unknown')}")
        print(f"   Score: {result.get('final_score', 0):.2f}")
    
    client.close()


if __name__ == '__main__':
    main()
