"""
Result Ranking Module

This module provides advanced ranking algorithms for quote search results.
"""
import logging
from typing import List, Dict
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuoteRanker:
    """
    Advanced ranking for quote search results.
    
    Combines multiple signals to rank quotes by relevance including
    full-text score, length match, prefix matching, and edit distance.
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize the ranker.
        
        Args:
            weights: Dictionary of ranking weights
        """
        self.weights = weights or {
            'search_score': 0.5,
            'length_match': 0.2,
            'prefix_bonus': 0.15,
            'edit_distance': 0.15
        }
    
    def rank(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Rank search results by relevance.
        
        Args:
            query: Original search query
            results: List of search results
            
        Returns:
            Ranked list of results with scores
        """
        query_lower = query.lower().strip()
        
        for result in results:
            quote = result.get('quote', '').lower().strip()
            
            # Calculate individual scores
            search_score = result.get('score', 0)
            length_score = self._length_match_score(query, quote)
            prefix_score = self._prefix_match_score(query_lower, quote)
            similarity_score = self._similarity_score(query_lower, quote)
            
            # Weighted combination
            final_score = (
                self.weights['search_score'] * search_score +
                self.weights['length_match'] * length_score +
                self.weights['prefix_bonus'] * prefix_score +
                self.weights['edit_distance'] * similarity_score
            )
            
            result['ranking_details'] = {
                'search_score': search_score,
                'length_score': length_score,
                'prefix_score': prefix_score,
                'similarity_score': similarity_score
            }
            result['final_score'] = final_score
        
        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return results
    
    def _length_match_score(self, query: str, quote: str) -> float:
        """
        Calculate length match score.
        
        Prefer quotes with similar length to the query.
        
        Args:
            query: Search query
            quote: Quote text
            
        Returns:
            Length match score (0-1)
        """
        query_len = len(query)
        quote_len = len(quote)
        
        if query_len == 0:
            return 0.0
        
        length_ratio = min(query_len, quote_len) / max(query_len, quote_len)
        return length_ratio
    
    def _prefix_match_score(self, query: str, quote: str) -> float:
        """
        Calculate prefix match score.
        
        Give bonus if quote starts with the query.
        
        Args:
            query: Search query
            quote: Quote text
            
        Returns:
            Prefix match score (0-1)
        """
        if not query:
            return 0.0
        
        if quote.startswith(query):
            return 1.0
        
        # Partial prefix match
        common_prefix_len = 0
        for q_char, quote_char in zip(query, quote):
            if q_char == quote_char:
                common_prefix_len += 1
            else:
                break
        
        return common_prefix_len / len(query)
    
    def _similarity_score(self, query: str, quote: str) -> float:
        """
        Calculate similarity score using edit distance.
        
        Args:
            query: Search query
            quote: Quote text
            
        Returns:
            Similarity score (0-1)
        """
        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(None, query, quote).ratio()
        return similarity


def main():
    """Test the ranker."""
    ranker = QuoteRanker()
    
    query = "to be or not to be"
    
    test_results = [
        {
            'quote': 'To be, or not to be, that is the question.',
            'author': 'Shakespeare',
            'score': 0.95
        },
        {
            'quote': 'To be or not to be is a famous quote.',
            'author': 'Unknown',
            'score': 0.85
        },
        {
            'quote': 'Being or not being, that is what matters.',
            'author': 'Someone',
            'score': 0.75
        }
    ]
    
    print(f"Query: '{query}'\n")
    ranked = ranker.rank(query, test_results)
    
    print("Ranked Results:")
    for i, result in enumerate(ranked, 1):
        print(f"\n{i}. {result['quote']}")
        print(f"   Author: {result['author']}")
        print(f"   Final Score: {result['final_score']:.3f}")
        print(f"   Details: {result['ranking_details']}")


if __name__ == '__main__':
    main()
