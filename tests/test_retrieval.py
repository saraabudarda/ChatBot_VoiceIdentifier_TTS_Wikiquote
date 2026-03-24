"""
Integration tests for Quote Retrieval
"""
import pytest
from src.chatbot.intent_recognizer import IntentRecognizer, Intent
from src.chatbot.response_generator import ResponseGenerator


class TestIntentRecognition:
    """Test cases for intent recognition."""
    
    def test_quote_completion_intent(self):
        """Test quote completion intent recognition."""
        recognizer = IntentRecognizer()
        
        # Quoted text
        result = recognizer.recognize('"To be or not to be"')
        assert result['intent'] == Intent.QUOTE_COMPLETION
        
        # Quote-like fragment
        result = recognizer.recognize('To be or not to be')
        assert result['intent'] == Intent.QUOTE_COMPLETION
    
    def test_attribution_intent(self):
        """Test attribution intent recognition."""
        recognizer = IntentRecognizer()
        
        result = recognizer.recognize("Who said 'I think therefore I am'?")
        assert result['intent'] == Intent.QUOTE_ATTRIBUTION
        assert 'quote' in result['entities']
    
    def test_find_by_author_intent(self):
        """Test find by author intent."""
        recognizer = IntentRecognizer()
        
        result = recognizer.recognize("Quotes by Shakespeare")
        assert result['intent'] == Intent.FIND_BY_AUTHOR
        
        result = recognizer.recognize("Show me Einstein quotes")
        assert result['intent'] == Intent.FIND_BY_AUTHOR
    
    def test_random_quote_intent(self):
        """Test random quote intent."""
        recognizer = IntentRecognizer()
        
        result = recognizer.recognize("Random quote")
        assert result['intent'] == Intent.RANDOM_QUOTE
        
        result = recognizer.recognize("Surprise me")
        assert result['intent'] == Intent.RANDOM_QUOTE
    
    def test_recommendation_intent(self):
        """Test recommendation intent."""
        recognizer = IntentRecognizer()
        
        result = recognizer.recognize("Quotes about courage")
        assert result['intent'] == Intent.QUOTE_RECOMMENDATION
        assert 'topic' in result['entities']


class TestResponseGeneration:
    """Test cases for response generation."""
    
    def test_completion_response(self):
        """Test quote completion response."""
        generator = ResponseGenerator()
        
        results = [{
            'quote': 'To be, or not to be, that is the question.',
            'author': 'William Shakespeare',
            'work': 'Hamlet'
        }]
        
        response = generator.generate(Intent.QUOTE_COMPLETION, results)
        
        assert 'To be, or not to be' in response
        assert 'Shakespeare' in response
        assert 'Hamlet' in response
    
    def test_attribution_response(self):
        """Test attribution response."""
        generator = ResponseGenerator()
        
        results = [{
            'quote': 'I think, therefore I am.',
            'author': 'René Descartes',
            'work': 'Discourse on the Method'
        }]
        
        response = generator.generate(Intent.QUOTE_ATTRIBUTION, results)
        
        assert 'Descartes' in response
        assert 'author' in response.lower() or 'by' in response.lower()
    
    def test_no_results_response(self):
        """Test no results response."""
        generator = ResponseGenerator()
        
        response = generator.generate(
            Intent.QUOTE_COMPLETION,
            [],
            {'partial_quote': 'test'}
        )
        
        assert "couldn't find" in response.lower() or "no" in response.lower()
    
    def test_multiple_results_response(self):
        """Test multiple results response."""
        generator = ResponseGenerator()
        
        results = [
            {'quote': 'Quote 1', 'author': 'Author 1', 'work': 'Work 1'},
            {'quote': 'Quote 2', 'author': 'Author 2', 'work': 'Work 2'},
            {'quote': 'Quote 3', 'author': 'Author 3', 'work': 'Work 3'},
        ]
        
        response = generator.generate(Intent.QUOTE_COMPLETION, results)
        
        # Should mention multiple results
        assert '1.' in response or 'found' in response.lower()


def test_intent_and_response_integration():
    """Test integration of intent recognition and response generation."""
    recognizer = IntentRecognizer()
    generator = ResponseGenerator()
    
    # Simulate user query
    user_input = "Who said 'to be or not to be'?"
    
    # Recognize intent
    intent_result = recognizer.recognize(user_input)
    assert intent_result['intent'] == Intent.QUOTE_ATTRIBUTION
    
    # Generate response (with mock results)
    mock_results = [{
        'quote': 'To be, or not to be, that is the question.',
        'author': 'William Shakespeare',
        'work': 'Hamlet'
    }]
    
    response = generator.generate(
        intent_result['intent'],
        mock_results,
        intent_result['entities']
    )
    
    assert 'Shakespeare' in response


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
