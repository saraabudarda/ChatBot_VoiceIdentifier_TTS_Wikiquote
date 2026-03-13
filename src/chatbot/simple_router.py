"""
Simple Router Module

Routes user queries to either:
1. AUTOCOMPLETE - for quote completion/search (90% of queries)
2. CLARIFICATION - for ambiguous queries that need user input

This replaces the complex intent recognition system with a simpler approach.
"""
import logging
import re
from typing import Tuple, Optional, Dict
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RouteType(Enum):
    """Route types for user queries."""
    AUTOCOMPLETE = "autocomplete"
    CLARIFICATION = "clarification"


class SimpleRouter:
    """
    Simple query router that determines if we can autocomplete or need clarification.
    
    Philosophy: Most queries are autocomplete requests. Only ask for clarification
    when the query is genuinely ambiguous.
    """
    
    def __init__(self):
        """Initialize router with ambiguous term patterns."""
        self.ambiguous_terms = [
            'the prophet', 'a prophet',
            'the philosopher', 'a philosopher', 
            'the poet', 'a poet',
            'the writer', 'a writer',
            'the author', 'an author',
            'the president', 'a president',
            'the king', 'a king',
            'the queen', 'a queen'
        ]
    
    def route(self, user_query: str) -> Tuple[RouteType, Optional[str], Dict]:
        """
        Route user query to autocomplete or clarification.
        
        Args:
            user_query: Raw user input
            
        Returns:
            Tuple of (route_type, clarification_message, metadata)
            - route_type: AUTOCOMPLETE or CLARIFICATION
            - clarification_message: Question to ask user (if CLARIFICATION)
            - metadata: Additional routing metadata
        """
        query_lower = user_query.lower().strip()
        
        # Check for ambiguous terms
        for term in self.ambiguous_terms:
            if term in query_lower:
                clarification = self._generate_clarification(term, user_query)
                return (RouteType.CLARIFICATION, clarification, {'ambiguous_term': term})
        
        # Check for empty/too short query
        if len(query_lower) < 3:
            clarification = ("**Query too short.** Please provide at least 3 characters.\n\n"
                           "Examples:\n"
                           "- \"to be or not to be\"\n"
                           "- \"quotes by Einstein\"\n"
                           "- \"imagination is more important\"")
            return (RouteType.CLARIFICATION, clarification, {'reason': 'too_short'})
        
        # Default: Route to autocomplete
        # The autocomplete engine will handle all variations:
        # - Quote completion: "to be or not to be"
        # - Author search: "quotes by Einstein", "Einstein quotes"
        # - Work search: "quotes from Hamlet"
        # - Topic search: "quotes about love"
        
        return (RouteType.AUTOCOMPLETE, None, {'query': user_query})
    
    def _generate_clarification(self, ambiguous_term: str, original_query: str) -> str:
        """
        Generate clarification question for ambiguous term.
        
        Args:
            ambiguous_term: The ambiguous term detected
            original_query: Original user query
            
        Returns:
            Clarification question with options
        """
        if 'prophet' in ambiguous_term:
            return ("**Which prophet do you mean?**\n\n"
                   "Please specify:\n"
                   "- Muhammad\n"
                   "- Moses\n"
                   "- Jesus\n"
                   "- Another specific prophet\n\n"
                   "Example: \"quotes by Muhammad\" or \"quotes by Prophet Muhammad\"")
        
        elif 'philosopher' in ambiguous_term:
            return ("**Which philosopher do you mean?**\n\n"
                   "Please specify a name, for example:\n"
                   "- Aristotle\n"
                   "- Plato\n"
                   "- Socrates\n"
                   "- Kant\n"
                   "- Nietzsche\n\n"
                   "Example: \"quotes by Aristotle\"")
        
        elif 'poet' in ambiguous_term:
            return ("**Which poet do you mean?**\n\n"
                   "Please specify a name, for example:\n"
                   "- Shakespeare\n"
                   "- Rumi\n"
                   "- Whitman\n"
                   "- Dickinson\n\n"
                   "Example: \"quotes by Rumi\"")
        
        elif 'president' in ambiguous_term:
            return ("**Which president do you mean?**\n\n"
                   "Please specify a name, for example:\n"
                   "- Abraham Lincoln\n"
                   "- George Washington\n"
                   "- Thomas Jefferson\n"
                   "- Barack Obama\n\n"
                   "Example: \"quotes by Abraham Lincoln\"")
        
        else:
            return (f"**'{ambiguous_term}' is too vague.**\n\n"
                   f"Please provide a specific name.\n\n"
                   f"Example: Instead of '{original_query}', try 'quotes by [specific name]'")


def main():
    """Test the simple router."""
    router = SimpleRouter()
    
    test_queries = [
        "to be or not to be",
        "quotes by Einstein",
        "find me a quote from the prophet",
        "imagination is more important",
        "quotes from Hamlet",
        "the philosopher said",
        "ab",  # too short
        "quotes about love"
    ]
    
    print("Testing Simple Router")
    print("=" * 80)
    
    for query in test_queries:
        route_type, clarification, metadata = router.route(query)
        print(f"\nQuery: \"{query}\"")
        print(f"Route: {route_type.value}")
        
        if clarification:
            print(f"Clarification:\n{clarification}")
        else:
            print(f"→ Send to autocomplete engine")
        print("-" * 80)


if __name__ == '__main__':
    main()
