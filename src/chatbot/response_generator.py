"""
Response Generator Module with Anti-Hallucination Rules

This module generates natural language responses from database
query results with strict verification to prevent hallucinations.

CRITICAL RULES:
1. Never assume an author - only use explicitly named authors
2. No stale context - don't reuse previous queries unless explicit
3. Ask clarification for ambiguous queries
4. Database-backed responses only - never invent quotes
5. Diversify results - avoid repeating same top 5
6. Explicit source attribution - always show Wikiquote page title
"""
import logging
from typing import List, Dict, Optional, Set
from .intent_recognizer import Intent
from .author_mapper import map_author, get_source_info, is_famous_work

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Natural language response generator with anti-hallucination controls.
    
    Converts database query results into user-friendly natural
    language responses with strict verification rules.
    """
    
    def __init__(self):
        """Initialize response generator with conversation state tracking."""
        self.shown_quotes: Set[str] = set()  # Track shown quotes to avoid repetition
        self.last_author: Optional[str] = None
        self.last_intent: Optional[Intent] = None
    
    def reset_conversation(self):
        """Reset conversation state (call when starting new chat session)."""
        self.shown_quotes.clear()
        self.last_author = None
        self.last_intent = None
    
    def _is_ambiguous_author(self, author_query: str) -> bool:
        """
        Check if author query is ambiguous and needs clarification.
        
        Args:
            author_query: Author name from user query
            
        Returns:
            True if ambiguous, False if specific
        """
        ambiguous_terms = [
            'the prophet', 'a prophet', 'the philosopher', 'a philosopher',
            'the poet', 'a poet', 'the writer', 'a writer', 'the author',
            'a author', 'the president', 'a president', 'the king', 'a king'
        ]
        
        author_lower = author_query.lower().strip()
        return any(term in author_lower for term in ambiguous_terms)
    
    def _generate_clarification(self, author_query: str) -> str:
        """
        Generate clarification question for ambiguous author.
        
        Args:
            author_query: Ambiguous author query
            
        Returns:
            Clarification question with options
        """
        author_lower = author_query.lower()
        
        if 'prophet' in author_lower:
            return ("**Clarification needed:** Which prophet do you mean?\n\n"
                   "Options:\n"
                   "- Muhammad (Islamic prophet)\n"
                   "- Moses (Biblical prophet)\n"
                   "- Jesus (Christian prophet)\n"
                   "- Other specific prophet?\n\n"
                   "Please specify the exact name.")
        
        elif 'philosopher' in author_lower:
            return ("**Clarification needed:** Which philosopher do you mean?\n\n"
                   "Examples: Aristotle, Plato, Socrates, Kant, Nietzsche, Descartes\n\n"
                   "Please specify the exact name.")
        
        elif 'poet' in author_lower:
            return ("**Clarification needed:** Which poet do you mean?\n\n"
                   "Examples: Shakespeare, Rumi, Whitman, Dickinson, Frost\n\n"
                   "Please specify the exact name.")
        
        else:
            return (f"**Clarification needed:** '{author_query}' is too ambiguous.\n\n"
                   "Please provide a specific name (e.g., 'Albert Einstein', 'Mark Twain').")
    
    def _filter_shown_quotes(self, results: List[Dict]) -> List[Dict]:
        """
        Filter out quotes already shown in this conversation.
        
        Args:
            results: List of quote results
            
        Returns:
            Filtered results excluding previously shown quotes
        """
        filtered = []
        for result in results:
            quote_text = result.get('quote', '')
            # Use first 100 chars as fingerprint
            fingerprint = quote_text[:100]
            if fingerprint not in self.shown_quotes:
                filtered.append(result)
        
        return filtered
    
    def _mark_quote_shown(self, quote_text: str):
        """Mark a quote as shown to avoid repetition."""
        fingerprint = quote_text[:100]
        self.shown_quotes.add(fingerprint)
    
    def generate(self, intent: Intent, results: List[Dict], entities: Dict = None) -> str:
        """
        Generate a natural language response with anti-hallucination controls.
        
        Args:
            intent: Recognized user intent
            results: Query results from database
            entities: Extracted entities from user input
            
        Returns:
            Natural language response string
        """
        entities = entities or {}
        
        # RULE 3: Check for ambiguous author queries
        if intent == Intent.FIND_BY_AUTHOR:
            author_query = entities.get('author', '')
            if self._is_ambiguous_author(author_query):
                return self._generate_clarification(author_query)
        
        # RULE 7: No results = honest response
        if not results:
            return self._generate_no_results_response(intent, entities)
        
        # RULE 5: Filter out previously shown quotes
        filtered_results = self._filter_shown_quotes(results)
        if not filtered_results:
            # All results were already shown
            return ("I've already shown you all the matching quotes I found. "
                   "Try a different query or ask for quotes from a different author.")
        
        # Route to appropriate response generator
        if intent == Intent.QUOTE_COMPLETION:
            return self._generate_completion_response(filtered_results)
        elif intent == Intent.QUOTE_ATTRIBUTION:
            return self._generate_attribution_response(filtered_results)
        elif intent == Intent.FIND_BY_AUTHOR:
            return self._generate_author_quotes_response(filtered_results, entities)
        elif intent == Intent.FIND_BY_WORK:
            return self._generate_work_quotes_response(filtered_results, entities)
        elif intent == Intent.RANDOM_QUOTE:
            return self._generate_random_quote_response(filtered_results)
        elif intent == Intent.QUOTE_RECOMMENDATION:
            return self._generate_recommendation_response(filtered_results, entities)
        else:
            return self._generate_general_response(filtered_results)
    
    def _generate_no_results_response(self, intent: Intent, entities: Dict) -> str:
        """
        Generate honest 'no results' response.
        
        RULE 7: Never invent quotes when database has no match.
        """
        if intent == Intent.FIND_BY_AUTHOR:
            author = entities.get('author', 'that author')
            return (f"**No results found** for '{author}' in the Wikiquote database.\n\n"
                   f"Suggestions:\n"
                   f"- Check the spelling\n"
                   f"- Try the full name (e.g., 'Albert Einstein' not 'Einstein')\n"
                   f"- This person may not have a Wikiquote page")
        
        elif intent == Intent.QUOTE_COMPLETION:
            return ("**I couldn't find a reliable match** in Wikiquote for that quote.\n\n"
                   "Try:\n"
                   "- A longer snippet (at least 5-6 words)\n"
                   "- Different phrasing\n"
                   "- Checking if the quote is actually from Wikiquote")
        
        else:
            return ("**No matching quotes found** in the database.\n\n"
                   "Try rephrasing your query or searching for a specific author.")
    
    def _generate_completion_response(self, results: List[Dict]) -> str:
        """
        Generate response for quote completion with source attribution.
        
        RULE 6: Always include source (Wikiquote page title).
        """
        if len(results) == 1:
            quote = results[0]
            original_author = quote.get('author', 'Unknown')
            work = quote.get('work')
            quote_text = quote.get('quote', '')
            
            # Mark as shown
            self._mark_quote_shown(quote_text)
            
            # Map to actual author
            actual_author, confidence = map_author(original_author, work)
            
            response = f"**Quote:** \"{quote_text}\"\n\n"
            response += f"**Source:** {actual_author}"
            
            if work and work != 'Unknown':
                response += f" (from *{work}*)"
            
            # Add confidence note if needed
            if confidence == 'medium':
                response += "\n\n*Note: Commonly attributed to this author*"
            elif confidence == 'low':
                response += "\n\n*Note: Attribution uncertain - verify independently*"
            
            return response
        else:
            response = f"**Found {len(results)} matching quotes:**\n\n"
            
            for i, quote in enumerate(results[:5], 1):
                quote_text = quote.get('quote', '')
                original_author = quote.get('author', 'Unknown')
                work = quote.get('work')
                actual_author, _ = map_author(original_author, work)
                
                # Mark as shown
                self._mark_quote_shown(quote_text)
                
                # Truncate long quotes
                display_quote = quote_text if len(quote_text) <= 150 else f"{quote_text[:150]}..."
                
                response += f"**{i}.** \"{display_quote}\"\n"
                response += f"   — {actual_author}"
                
                if work and work != 'Unknown':
                    response += f", *{work}*"
                
                response += "\n\n"
            
            return response
    
    def _generate_attribution_response(self, results: List[Dict]) -> str:
        """
        Generate response for quote attribution.
        
        RULE 6: Explicit source attribution format.
        """
        quote_data = results[0]
        original_author = quote_data.get('author', 'Unknown')
        work = quote_data.get('work')
        quote_text = quote_data.get('quote', '')
        
        # Mark as shown
        self._mark_quote_shown(quote_text)
        
        # Map work/character name to actual author
        actual_author, confidence = map_author(original_author, work)
        
        # Get source information
        source_info = get_source_info(original_author, work)
        
        # Build response in structured format
        response = f"**Author:** {actual_author}\n\n"
        
        if source_info and source_info != "Source unknown":
            response += f"**Source:** {source_info}\n\n"
        
        response += f"**Quote:** \"{quote_text}\"\n\n"
        
        # Add confidence note if not high
        if confidence == 'medium':
            response += "*Note: This quote is commonly attributed to this author.*\n"
        elif confidence == 'low':
            response += "*Note: Attribution uncertain. Please verify independently.*\n"
        
        return response
    
    def _generate_author_quotes_response(self, results: List[Dict], entities: Dict) -> str:
        """
        Generate response for author quotes search.
        
        RULE 1: Only if author was explicitly named.
        """
        requested_author = entities.get('author', 'this author')
        
        # Map the first result's author to get the actual author name
        if results:
            first_result = results[0]
            actual_author, _ = map_author(first_result.get('author', requested_author), first_result.get('work'))
            response = f"**Quotes by {actual_author}:**\n\n"
        else:
            response = f"**Quotes by {requested_author}:**\n\n"
        
        for i, quote in enumerate(results[:5], 1):
            quote_text = quote.get('quote', '')
            work = quote.get('work')
            
            # Mark as shown
            self._mark_quote_shown(quote_text)
            
            response += f"**{i}.** \"{quote_text}\"\n"
            
            if work and work != 'Unknown':
                response += f"   *Source: {work}*\n"
            
            response += "\n"
        
        if len(results) > 5:
            response += f"\n*({len(results) - 5} more quotes available)*"
        
        return response
    
    def _generate_work_quotes_response(self, results: List[Dict], entities: Dict) -> str:
        """Generate response for work quotes search."""
        work = entities.get('work', 'this work')
        
        response = f"**Quotes from** *{work}*:\n\n"
        
        for i, quote in enumerate(results[:5], 1):
            quote_text = quote.get('quote', '')
            
            # Mark as shown
            self._mark_quote_shown(quote_text)
            
            response += f"**{i}.** \"{quote_text}\"\n\n"
        
        if len(results) > 5:
            response += f"\n*({len(results) - 5} more quotes available)*"
        
        return response
    
    def _generate_random_quote_response(self, results: List[Dict]) -> str:
        """Generate response for random quote request."""
        if not results:
            return "No quotes available."
        
        quote = results[0]
        quote_text = quote.get('quote', '')
        original_author = quote.get('author', 'Unknown')
        work = quote.get('work')
        
        # Mark as shown
        self._mark_quote_shown(quote_text)
        
        actual_author, _ = map_author(original_author, work)
        
        response = f"**Random Quote:**\n\n\"{quote_text}\"\n\n"
        response += f"— **{actual_author}**"
        
        if work and work != 'Unknown':
            response += f", *{work}*"
        
        return response
    
    def _generate_recommendation_response(self, results: List[Dict], entities: Dict) -> str:
        """Generate response for quote recommendations."""
        topic = entities.get('topic', 'this topic')
        
        response = f"**Quotes about** *{topic}*:\n\n"
        
        for i, quote in enumerate(results[:5], 1):
            quote_text = quote.get('quote', '')
            original_author = quote.get('author', 'Unknown')
            work = quote.get('work')
            actual_author, _ = map_author(original_author, work)
            
            # Mark as shown
            self._mark_quote_shown(quote_text)
            
            response += f"**{i}.** \"{quote_text}\"\n"
            response += f"   — {actual_author}\n\n"
        
        return response
    
    def _generate_general_response(self, results: List[Dict]) -> str:
        """Generate general response for unclassified queries."""
        response = "**Search Results:**\n\n"
        
        for i, quote in enumerate(results[:5], 1):
            quote_text = quote.get('quote', '')
            original_author = quote.get('author', 'Unknown')
            work = quote.get('work')
            actual_author, _ = map_author(original_author, work)
            
            # Mark as shown
            self._mark_quote_shown(quote_text)
            
            display_quote = quote_text if len(quote_text) <= 150 else f"{quote_text[:150]}..."
            
            response += f"**{i}.** \"{display_quote}\"\n"
            response += f"   — {actual_author}"
            
            if work and work != 'Unknown':
                response += f", *{work}*"
            
            response += "\n\n"
        
        return response
