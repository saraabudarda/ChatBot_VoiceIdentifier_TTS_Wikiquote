"""
Quote Autocompletion Engine

This module implements quote retrieval and autocompletion using
Neo4j full-text search with ranking and relevance scoring.
"""
import logging
from typing import List, Dict, Optional
import unicodedata
import difflib
from ..database.neo4j_client import Neo4jClient
from ..database.indexing import IndexManager

def _normalize_name(name: str) -> str:
    """Normalize author name (lowercase, remove accents, strip spaces)."""
    if not name: return ""
    name = name.lower()
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    return ' '.join(name.split())


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
        
        # Load author cache for fast fuzzy matching
        self._author_cache = {}
        self._load_author_cache()
    
    def _load_author_cache(self):
        query = "MATCH (p:Person) RETURN p.name AS name"
        try:
            results = self.client.execute_query(query)
            for r in results:
                name = r.get('name')
                if name:
                    self._author_cache[_normalize_name(name)] = name
            logger.info(f"Loaded {len(self._author_cache)} authors into fuzzy match cache.")
        except Exception as e:
            logger.error(f"Failed to load author cache: {e}")

    def _resolve_author_name(self, input_name: str) -> tuple[Optional[str], str, float]:
        """Resolves author name using exact, normalized, then fuzzy matching."""
        norm_input = _normalize_name(input_name)
        if not norm_input:
            return None, "none", 0.0
            
        # 1. Exact Normalized Match
        if norm_input in self._author_cache:
            return self._author_cache[norm_input], "exact_normalized", 1.0
            
        # 2. Fuzzy Match
        matches = difflib.get_close_matches(norm_input, self._author_cache.keys(), n=1, cutoff=0.7)
        if matches:
            best_match_norm = matches[0]
            conf = difflib.SequenceMatcher(None, norm_input, best_match_norm).ratio()
            return self._author_cache[best_match_norm], "fuzzy", conf
            
        return None, "none", 0.0
    
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
        
        # Fetch MORE results from database than requested (to ensure famous quotes aren't filtered out)
        # We'll rank them and return the top max_results
        fetch_limit = max(50, max_results * 5)  # Fetch at least 50 or 5x requested
        
        # Search using full-text index
        results = self.index_manager.search_quotes(partial_quote, limit=fetch_limit)

        # Deduplicate — only remove exact (text + author) duplicates.
        # Same quote text attributed to DIFFERENT authors = valid distinct results, keep both.
        seen = set()
        unique_results = []
        for r in results:
            text_key = r.get('quote', '').strip().lower()[:120]
            author_key = r.get('author', '').strip().lower()
            key = (text_key, author_key)
            if text_key and key not in seen:
                seen.add(key)
                unique_results.append(r)
        results = unique_results

        # Enhance results with ranking
        ranked_results = self._rank_results(partial_quote, results)

        # Return only the top max_results after ranking
        return ranked_results[:max_results]
    
    def find_by_author(self, author_name: str, limit: int = 10) -> List[Dict]:
        """
        Find quotes by a specific author with quality filtering.
        
        Filters out:
        - Short metadata entries (< 50 chars)
        - Citations with years (1797)
        - Page references (p. 338)
        - URLs
        - Incomplete text (&c.)
        
        Args:
            author_name: Name of the author
            limit: Maximum number of results
            
        Returns:
            List of high-quality quotes by the author
        """
        # Step 1: Resolve the author name robustly
        resolved_name, match_type, conf = self._resolve_author_name(author_name)
        
        if not resolved_name:
            # If no author could be found even with fuzzy matching, fallback fails gracefully
            logger.info(f"find_by_author: '{author_name}' could not be resolved.")
            return []
            
        logger.info(f"find_by_author: Resolved '{author_name}' -> '{resolved_name}' ({match_type}, conf={conf:.2f})")
        
        # We now search for the EXACT resolved name in the database
        query = """
        MATCH (p:Person {name: $resolved_name})-[:SAID]->(q:Quote)
        WHERE q.quality_bucket IN ['high_quality', 'review']
          AND coalesce(q.is_canonical, true) = true
          AND size(q.text) >= 40
          AND NOT q.text =~ '.*\\(\\d{4}\\).*'
          AND NOT q.text =~ '.*p\\. \\d+.*'
          AND NOT q.text =~ '.*https?://.*'
          AND NOT q.text STARTS WITH 'http'
          AND NOT q.text CONTAINS '&c.'
          AND NOT q.text =~ '.*ASIN:.*'
          AND NOT q.text =~ '.*published in.*'
        OPTIONAL MATCH (w:Work)-[:HAS_QUOTE]->(q)
        RETURN q.text AS quote,
               p.name AS author,
               w.name AS work,
               q.quality_score AS score
        ORDER BY q.quality_score DESC, size(q.text) ASC
        LIMIT $limit
        """
        
        params = {'resolved_name': resolved_name, 'limit': limit}
        results = self.client.execute_query(query, params)
        
        # Inject our debug info
        for r in results:
            r['_debug_author_match'] = {
                'original_query': author_name,
                'interpreted_author': resolved_name,
                'match_type': match_type,
                'confidence': conf
            }
        
        return results
    
    def find_by_work(self, work_title: str, limit: int = 10) -> List[Dict]:
        """
        Find quotes from a specific work using the new quality tiers.
        """
        query = """
        MATCH (w:Work)-[:HAS_QUOTE]->(q:Quote)
        WHERE toLower(w.name) CONTAINS toLower($work_title)
          AND q.quality_bucket IN ['high_quality', 'review']
          AND coalesce(q.is_canonical, true) = true
          AND size(q.text) >= 40
          AND NOT q.text =~ '.*\\(\\d{4}\\).*'
          AND NOT q.text =~ '.*p\\. \\d+.*'
          AND NOT q.text =~ '.*https?://.*'
          AND NOT q.text STARTS WITH 'http'
          AND NOT q.text CONTAINS '&c.'
          AND NOT q.text =~ '.*ASIN:.*'
          AND NOT q.text =~ '.*published in.*'
        OPTIONAL MATCH (p:Person)-[:SAID]->(q)
        RETURN q.text AS quote,
               coalesce(p.name, 'Unknown') AS author,
               w.name AS work,
               q.quality_score AS score
        ORDER BY q.quality_score DESC, size(q.text) ASC
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
        MATCH (p:Person)-[:SAID]->(q:Quote)
        WHERE q.quality_bucket = 'high_quality'
          AND coalesce(q.is_canonical, true) = true
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
            # List of famous authors/works that should be prioritized
            famous_authors = {
                'william shakespeare', 'shakespeare', 'hamlet', 'macbeth', 'romeo and juliet',
                'albert einstein', 'einstein', 'mark twain', 'oscar wilde',
                'aristotle', 'plato', 'socrates', 'confucius',
                'martin luther king', 'nelson mandela', 'mahatma gandhi',
                'winston churchill', 'abraham lincoln', 'benjamin franklin'
            }
            
            famous_works = {
                'hamlet', 'macbeth', 'romeo and juliet', 'othello', 'king lear',
                'the prophet', 'the bible', 'the quran', 'the odyssey', 'the iliad'
            }
            
            author_lower = author.lower() if author else ''
            work_lower = work.lower() if work else ''
            
            # Check if author or work is famous
            is_famous = any(famous in author_lower for famous in famous_authors) or \
                       any(famous in work_lower for famous in famous_works)
            
            if is_famous:
                # Famous author/work - MAJOR boost
                author_boost = 4.0  # Increased to prioritize famous quotes
            elif author and author != 'Unknown' and len(author) > 2:
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
            # Reduce prefix bonus to not overwhelm famous author boost
            if quote_lower.startswith(query_lower):
                prefix_bonus = 1.2  # Further reduced
            elif query_lower in quote_lower[:50]:
                # Query appears in first 50 chars
                prefix_bonus = 1.1  # Further reduced
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
                base_score * 0.2 +          # Full-text relevance (20%)
                length_score * 0.1 +         # Length preference (10%)
                author_boost * 0.5 +         # Author quality (50%, MAJOR boost for famous)
                work_boost * 0.1 +           # Work attribution (10%)
                coverage_score * 0.1         # Query coverage (10%)
            ) * prefix_bonus                 # Prefix match multiplier
            
            result['final_score'] = final_score
            result['_debug'] = {
                'base': base_score,
                'length': length_score,
                'author': author_boost,
                'work': work_boost,
                'prefix': prefix_bonus,
                'coverage': coverage_score,
                'is_famous': is_famous
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
