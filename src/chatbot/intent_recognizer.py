"""
Intent Recognition Module

This module classifies user intents from natural language input
to determine the appropriate action for the chatbot.
"""
import logging
import re
from typing import Dict, Optional
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Intent(Enum):
    """Enumeration of supported user intents."""
    QUOTE_COMPLETION = "quote_completion"
    QUOTE_ATTRIBUTION = "quote_attribution"
    QUOTE_RECOMMENDATION = "quote_recommendation"
    FIND_BY_AUTHOR = "find_by_author"
    FIND_BY_WORK = "find_by_work"
    RANDOM_QUOTE = "random_quote"
    GENERAL_QUERY = "general_query"


class IntentRecognizer:
    """
    Rule-based intent classification for quote-related queries.
    
    Identifies user intent from natural language input using
    pattern matching and keyword detection.
    """
    
    # Intent patterns
    PATTERNS = {
        Intent.QUOTE_ATTRIBUTION: [
            r"who (said|wrote|authored|spoke)",
            r"who('s| is) the author",
            r"author of",
            r"who quoted",
            r"source of (the )?quote",
        ],
        Intent.FIND_BY_AUTHOR: [
            r"quotes? (by|from) ([A-Z][a-z]+ ?)+",
            r"([A-Z][a-z]+ ?)+ quotes?",
            r"show me .+ quotes?",
            r"find quotes? (by|from)",
        ],
        Intent.FIND_BY_WORK: [
            r"quotes? from (.+)",
            r"in (the )?(book|movie|play|speech)",
            r"from (the )?(book|movie|play|speech)",
        ],
        Intent.RANDOM_QUOTE: [
            r"random quote",
            r"surprise me",
            r"any quote",
            r"give me (a )?quote",
        ],
        Intent.QUOTE_RECOMMENDATION: [
            r"quotes? about",
            r"find quotes? (on|regarding)",
            r"recommend .+ quotes?",
            r"quotes? related to",
        ],
    }
    
    def __init__(self):
        """Initialize the intent recognizer."""
        # Compile patterns for efficiency
        self.compiled_patterns = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in self.PATTERNS.items()
        }
    
    def recognize(self, user_input: str) -> Dict:
        """
        Recognize intent from user input.
        
        Args:
            user_input: User's natural language input
            
        Returns:
            Dictionary with intent and extracted entities
        """
        user_input = user_input.strip()
        
        if not user_input:
            return {'intent': Intent.GENERAL_QUERY, 'entities': {}}
        
        # Check for quote completion (contains quotes or looks like a quote fragment)
        if self._is_quote_completion(user_input):
            return {
                'intent': Intent.QUOTE_COMPLETION,
                'entities': {'partial_quote': self._extract_quote(user_input)}
            }
        
        # Check other patterns
        for intent, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(user_input)
                if match:
                    entities = self._extract_entities(intent, user_input, match)
                    return {'intent': intent, 'entities': entities}
        
        # Default to general query
        return {
            'intent': Intent.GENERAL_QUERY,
            'entities': {'query': user_input}
        }
    
    def _is_quote_completion(self, text: str) -> bool:
        """
        Check if input looks like a partial quote.
        
        Args:
            text: User input
            
        Returns:
            True if likely a quote completion request
        """
        # Contains quoted text
        if '"' in text or "'" in text or '"' in text or '"' in text:
            return True
        
        # Starts with common quote beginnings
        quote_starters = [
            'to be', 'i think', 'the only', 'life is', 'love is',
            'all that', 'it is', 'we are', 'you are', 'i am'
        ]
        
        text_lower = text.lower()
        for starter in quote_starters:
            if text_lower.startswith(starter):
                return True
        
        # Contains ellipsis or incomplete sentence markers
        if '...' in text or text.endswith(('...', '…')):
            return True
        
        # Short text without question words (likely a quote fragment)
        question_words = ['who', 'what', 'where', 'when', 'why', 'how', 'find', 'show', 'give']
        if len(text.split()) <= 10 and not any(text.lower().startswith(qw) for qw in question_words):
            return True
        
        return False
    
    def _extract_quote(self, text: str) -> str:
        """
        Extract quote text from input.
        
        Args:
            text: User input
            
        Returns:
            Extracted quote text
        """
        # Remove quotes if present
        text = text.strip('"\'""''')
        
        # Remove ellipsis
        text = text.replace('...', '').replace('…', '')
        
        return text.strip()
    
    def _extract_entities(self, intent: Intent, text: str, match: re.Match) -> Dict:
        """
        Extract entities based on intent.
        
        Args:
            intent: Recognized intent
            text: User input
            match: Regex match object
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {}
        
        if intent == Intent.QUOTE_ATTRIBUTION:
            # Extract the quote being asked about
            quote_match = re.search(r'["\'](.+?)["\']', text)
            if quote_match:
                entities['quote'] = quote_match.group(1)
            else:
                # Try to extract after "who said"
                after_match = re.search(r'who (?:said|wrote) (.+)', text, re.IGNORECASE)
                if after_match:
                    entities['quote'] = after_match.group(1).strip()
        
        elif intent == Intent.FIND_BY_AUTHOR:
            # Extract author name
            author_match = re.search(r'(?:by|from) ([A-Z][a-z]+(?: [A-Z][a-z]+)*)', text)
            if author_match:
                entities['author'] = author_match.group(1)
            else:
                # Try to find capitalized name
                name_match = re.search(r'([A-Z][a-z]+(?: [A-Z][a-z]+)+)', text)
                if name_match:
                    entities['author'] = name_match.group(1)
        
        elif intent == Intent.FIND_BY_WORK:
            # Extract work title
            work_match = re.search(r'from (?:the )?(.+)', text, re.IGNORECASE)
            if work_match:
                entities['work'] = work_match.group(1).strip()
        
        elif intent == Intent.QUOTE_RECOMMENDATION:
            # Extract topic
            topic_match = re.search(r'about (.+)', text, re.IGNORECASE)
            if topic_match:
                entities['topic'] = topic_match.group(1).strip()
        
        return entities


def main():
    """Test the intent recognizer."""
    recognizer = IntentRecognizer()
    
    test_inputs = [
        "To be or not to be",
        "Who said 'I think therefore I am'?",
        "Show me quotes by Shakespeare",
        "Quotes about courage",
        "Random quote please",
        "Quotes from Hamlet",
        "What is the meaning of life?",
    ]
    
    print("Testing Intent Recognition:\n")
    for user_input in test_inputs:
        result = recognizer.recognize(user_input)
        print(f"Input: {user_input}")
        print(f"Intent: {result['intent'].value}")
        print(f"Entities: {result['entities']}")
        print()


if __name__ == '__main__':
    main()
