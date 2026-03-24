"""
Unit tests for Text Cleaner
"""
import pytest
from src.ingestion.text_cleaner import TextCleaner


class TestTextCleaner:
    """Test cases for TextCleaner class."""
    
    def test_html_entity_decoding(self):
        """Test HTML entity decoding."""
        cleaner = TextCleaner()
        
        text = "This is a &quot;test&quot; with &nbsp;entities"
        cleaned = cleaner._decode_html_entities(text)
        
        assert '"test"' in cleaned
        assert '&nbsp;' not in cleaned
    
    def test_url_removal(self):
        """Test URL removal."""
        cleaner = TextCleaner()
        
        text = "Check out https://example.com for more info"
        cleaned = cleaner._remove_urls(text)
        
        assert 'https://example.com' not in cleaned
        assert 'Check out' in cleaned
    
    def test_unicode_normalization(self):
        """Test Unicode normalization."""
        cleaner = TextCleaner()
        
        text = "café naïve"
        normalized = cleaner._normalize_unicode(text)
        
        # Should normalize but preserve accents
        assert 'caf' in normalized
    
    def test_quote_normalization(self):
        """Test quote character normalization."""
        cleaner = TextCleaner()
        
        text = ""fancy quotes" and 'apostrophes'"
        normalized = cleaner._normalize_quotes(text)
        
        # Should convert to standard quotes
        assert '"fancy quotes"' in normalized or "'fancy quotes'" in normalized
    
    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        cleaner = TextCleaner()
        
        text = "Multiple   spaces    here"
        normalized = cleaner._normalize_whitespace(text)
        
        assert '   ' not in normalized
        assert 'Multiple spaces here' in normalized
    
    def test_deduplication(self):
        """Test duplicate detection."""
        cleaner = TextCleaner()
        
        # First occurrence should pass
        result1 = cleaner.clean("This is a unique quote")
        assert result1 is not None
        
        # Exact duplicate should be filtered
        result2 = cleaner.clean("This is a unique quote")
        assert result2 is None
    
    def test_near_duplicate_detection(self):
        """Test near-duplicate detection."""
        cleaner = TextCleaner(dedup_threshold=0.95)
        
        # First quote
        result1 = cleaner.clean("This is a test quote")
        assert result1 is not None
        
        # Very similar quote (should be filtered)
        result2 = cleaner.clean("This is a test quote!")
        assert result2 is None
    
    def test_short_text_filtering(self):
        """Test filtering of very short text."""
        cleaner = TextCleaner()
        
        # Too short
        result = cleaner.clean("Hi")
        assert result is None
        
        # Long enough
        result = cleaner.clean("This is long enough to be kept")
        assert result is not None


def test_cleaner_initialization():
    """Test cleaner initialization."""
    cleaner = TextCleaner(language_filter='en', dedup_threshold=0.9)
    
    assert cleaner.language_filter == 'en'
    assert cleaner.dedup_threshold == 0.9


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
