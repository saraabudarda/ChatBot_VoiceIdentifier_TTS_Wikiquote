"""
Response Generator Module

This module generates natural language responses from database
query results for the chatbot interface.
"""
import logging
from typing import List, Dict, Optional
from .intent_recognizer import Intent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Natural language response generator for the chatbot.
    
    Converts database query results into user-friendly natural
    language responses based on the recognized intent.
    """
    
    def generate(self, intent: Intent, results: List[Dict], entities: Dict = None) -> str:
        """
        Generate a natural language response.
        
        Args:
            intent: Recognized user intent
            results: Query results from database
            entities: Extracted entities from user input
            
        Returns:
            Natural language response string
        """
        entities = entities or {}
        
        if not results:
            return self._generate_no_results_response(intent, entities)
        
        if intent == Intent.QUOTE_COMPLETION:
            return self._generate_completion_response(results)
        elif intent == Intent.QUOTE_ATTRIBUTION:
            return self._generate_attribution_response(results)
        elif intent == Intent.FIND_BY_AUTHOR:
            return self._generate_author_quotes_response(results, entities)
        elif intent == Intent.FIND_BY_WORK:
            return self._generate_work_quotes_response(results, entities)
        elif intent == Intent.RANDOM_QUOTE:
            return self._generate_random_quote_response(results)
        elif intent == Intent.QUOTE_RECOMMENDATION:
            return self._generate_recommendation_response(results, entities)
        else:
            return self._generate_general_response(results)
    
    def _generate_completion_response(self, results: List[Dict]) -> str:
        """Generate response for quote completion."""
        if len(results) == 1:
            quote = results[0]
            response = f"Here's the complete quote:\n\n"
            response += f'"{quote["quote"]}"\n\n'
            response += f"— **{quote['author']}**"
            
            if quote.get('work') and quote['work'] != 'Unknown':
                response += f", from *{quote['work']}*"
            
            return response
        else:
            response = f"I found {len(results)} matching quotes. Here are the top results:\n\n"
            
            for i, quote in enumerate(results[:5], 1):
                response += f"**{i}.** \"{quote['quote'][:100]}{'...' if len(quote['quote']) > 100 else ''}\"\n"
                response += f"   — {quote['author']}"
                
                if quote.get('work') and quote['work'] != 'Unknown':
                    response += f", *{quote['work']}*"
                
                response += "\n\n"
            
            return response
    
    def _generate_attribution_response(self, results: List[Dict]) -> str:
        """Generate response for quote attribution."""
        quote = results[0]
        
        response = f"That quote is by **{quote['author']}**"
        
        if quote.get('work') and quote['work'] != 'Unknown':
            response += f", from *{quote['work']}*"
        
        response += ".\n\n"
        response += f'Full quote: "{quote["quote"]}"'
        
        return response
    
    def _generate_author_quotes_response(self, results: List[Dict], entities: Dict) -> str:
        """Generate response for author quotes search."""
        author = entities.get('author', 'this author')
        
        response = f"Here are some quotes by **{author}**:\n\n"
        
        for i, quote in enumerate(results[:5], 1):
            response += f"**{i}.** \"{quote['quote']}\"\n"
            
            if quote.get('work') and quote['work'] != 'Unknown':
                response += f"   — from *{quote['work']}*\n"
            
            response += "\n"
        
        if len(results) > 5:
            response += f"\n*({len(results) - 5} more quotes available)*"
        
        return response
    
    def _generate_work_quotes_response(self, results: List[Dict], entities: Dict) -> str:
        """Generate response for work quotes search."""
        work = entities.get('work', 'this work')
        
        response = f"Here are quotes from *{work}*:\n\n"
        
        for i, quote in enumerate(results[:5], 1):
            response += f"**{i}.** \"{quote['quote']}\"\n"
            response += f"   — {quote['author']}\n\n"
        
        if len(results) > 5:
            response += f"\n*({len(results) - 5} more quotes available)*"
        
        return response
    
    def _generate_random_quote_response(self, results: List[Dict]) -> str:
        """Generate response for random quote."""
        quote = results[0]
        
        response = f'"{quote["quote"]}"\n\n'
        response += f"— **{quote['author']}**"
        
        if quote.get('work') and quote['work'] != 'Unknown':
            response += f", from *{quote['work']}*"
        
        return response
    
    def _generate_recommendation_response(self, results: List[Dict], entities: Dict) -> str:
        """Generate response for quote recommendations."""
        topic = entities.get('topic', 'this topic')
        
        response = f"Here are some quotes about **{topic}**:\n\n"
        
        for i, quote in enumerate(results[:5], 1):
            response += f"**{i}.** \"{quote['quote']}\"\n"
            response += f"   — {quote['author']}"
            
            if quote.get('work') and quote['work'] != 'Unknown':
                response += f", *{quote['work']}*"
            
            response += "\n\n"
        
        return response
    
    def _generate_general_response(self, results: List[Dict]) -> str:
        """Generate general response."""
        if len(results) == 1:
            quote = results[0]
            return f'"{quote["quote"]}"\n\n— **{quote["author"]}**'
        else:
            response = f"I found {len(results)} results:\n\n"
            
            for i, quote in enumerate(results[:3], 1):
                response += f"**{i}.** \"{quote['quote'][:80]}...\"\n"
                response += f"   — {quote['author']}\n\n"
            
            return response
    
    def _generate_no_results_response(self, intent: Intent, entities: Dict) -> str:
        """Generate response when no results are found."""
        if intent == Intent.QUOTE_COMPLETION:
            return "I couldn't find that quote in my database. Could you provide more context or try a different phrase?"
        elif intent == Intent.QUOTE_ATTRIBUTION:
            quote = entities.get('quote', 'that quote')
            return f"I couldn't find the author of \"{quote}\" in my database. It might be paraphrased or from a less common source."
        elif intent == Intent.FIND_BY_AUTHOR:
            author = entities.get('author', 'that author')
            return f"I couldn't find any quotes by {author} in my database."
        elif intent == Intent.FIND_BY_WORK:
            work = entities.get('work', 'that work')
            return f"I couldn't find any quotes from {work} in my database."
        else:
            return "I couldn't find any matching quotes. Try rephrasing your query or searching for a different topic."


def main():
    """Test the response generator."""
    generator = ResponseGenerator()
    
    # Test completion response
    test_results = [
        {
            'quote': 'To be, or not to be, that is the question.',
            'author': 'William Shakespeare',
            'work': 'Hamlet',
            'score': 0.95
        }
    ]
    
    print("Testing Response Generator:\n")
    print("1. Quote Completion:")
    print(generator.generate(Intent.QUOTE_COMPLETION, test_results))
    print("\n" + "="*60 + "\n")
    
    print("2. Quote Attribution:")
    print(generator.generate(Intent.QUOTE_ATTRIBUTION, test_results))
    print("\n" + "="*60 + "\n")
    
    print("3. No Results:")
    print(generator.generate(Intent.QUOTE_COMPLETION, [], {'partial_quote': 'test'}))


if __name__ == '__main__':
    main()
