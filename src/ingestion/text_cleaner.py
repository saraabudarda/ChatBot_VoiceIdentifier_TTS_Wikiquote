"""
Text Cleaning Pipeline

This module provides deterministic text cleaning and normalization
for quote text extracted from Wikiquote.
"""
import re
import html
import unicodedata
import logging
from typing import List, Set, Optional
from difflib import SequenceMatcher

try:
    from langdetect import detect, LangDetectException
except ImportError:
    detect = None
    LangDetectException = Exception

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Deterministic text cleaning pipeline for quote normalization.
    
    Handles wiki markup removal, HTML entities, Unicode normalization,
    language filtering, and deduplication.
    """
    
    # Patterns for cleaning
    WIKI_MARKUP_PATTERN = re.compile(r'\[\[|\]\]|\{\{|\}\}')
    HTML_ENTITY_PATTERN = re.compile(r'&[a-zA-Z]+;|&#\d+;')
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    FOOTNOTE_PATTERN = re.compile(r'\[\d+\]|\[citation needed\]|\[note \d+\]', re.IGNORECASE)
    MULTIPLE_SPACES = re.compile(r'\s+')
    QUOTE_CHARS = {
        '"': '"', '"': '"', ''': "'", ''': "'",
        '«': '"', '»': '"', '‹': "'", '›': "'"
    }
    
    def __init__(self, language_filter: str = 'en', dedup_threshold: float = 0.95):
        """
        Initialize the text cleaner.
        
        Args:
            language_filter: Language code to keep (e.g., 'en' for English)
            dedup_threshold: Similarity threshold for near-duplicate detection (0-1)
        """
        self.language_filter = language_filter
        self.dedup_threshold = dedup_threshold
        self.seen_quotes: Set[str] = set()  # For exact deduplication
        self.seen_normalized: List[str] = []  # For fuzzy deduplication
        
    def clean(self, text: str) -> Optional[str]:
        """
        Apply full cleaning pipeline to text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text or None if text should be filtered out
        """
        if not text or len(text.strip()) < 10:
            return None
        
        # Step 1: Remove wiki markup remnants
        text = self._remove_wiki_markup(text)
        
        # Step 2: Decode HTML entities
        text = self._decode_html_entities(text)
        
        # Step 3: Remove URLs
        text = self._remove_urls(text)
        
        # Step 4: Remove footnotes
        text = self._remove_footnotes(text)
        
        # Step 5: Unicode normalization
        text = self._normalize_unicode(text)
        
        # Step 6: Normalize quotes and apostrophes
        text = self._normalize_quotes(text)
        
        # Step 7: Normalize whitespace
        text = self._normalize_whitespace(text)
        
        # Step 8: Language filtering
        if not self._is_target_language(text):
            return None
        
        # Step 9: Check for duplicates
        if self._is_duplicate(text):
            return None
        
        return text.strip()
    
    def _remove_wiki_markup(self, text: str) -> str:
        """Remove remaining wiki markup characters."""
        text = self.WIKI_MARKUP_PATTERN.sub('', text)
        # Remove other wiki syntax
        text = re.sub(r"'{2,5}", '', text)  # Bold/italic
        text = re.sub(r'<[^>]+>', '', text)  # HTML tags
        return text
    
    def _decode_html_entities(self, text: str) -> str:
        """Decode HTML entities like &nbsp;, &quot;, etc."""
        # First use html.unescape for standard entities
        text = html.unescape(text)
        # Handle numeric entities
        text = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), text)
        return text
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return self.URL_PATTERN.sub('', text)
    
    def _remove_footnotes(self, text: str) -> str:
        """Remove footnote markers and citations."""
        return self.FOOTNOTE_PATTERN.sub('', text)
    
    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode using NFKC normalization.
        
        NFKC: Compatibility decomposition followed by canonical composition.
        This handles different representations of the same character.
        """
        return unicodedata.normalize('NFKC', text)
    
    def _normalize_quotes(self, text: str) -> str:
        """Normalize various quote and apostrophe characters to standard ASCII."""
        for old_char, new_char in self.QUOTE_CHARS.items():
            text = text.replace(old_char, new_char)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace:
        - Replace multiple spaces with single space
        - Remove leading/trailing whitespace
        - Normalize line breaks
        """
        # Replace tabs and multiple spaces
        text = self.MULTIPLE_SPACES.sub(' ', text)
        # Clean up around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
        return text.strip()
    
    def _is_target_language(self, text: str) -> bool:
        """
        Check if text is in the target language.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is in target language, False otherwise
        """
        if detect is None:
            # If langdetect not available, assume all text is valid
            logger.warning("langdetect not available, skipping language filtering")
            return True
        
        try:
            # Detect language
            detected_lang = detect(text)
            return detected_lang == self.language_filter
        except (LangDetectException, Exception) as e:
            # If detection fails, be permissive and keep the text
            logger.debug(f"Language detection failed for text: {text[:50]}... Error: {e}")
            return True
    
    def _is_duplicate(self, text: str) -> bool:
        """
        Check if text is an exact or near-duplicate.
        
        Args:
            text: Text to check
            
        Returns:
            True if duplicate, False otherwise
        """
        # Exact duplicate check
        text_lower = text.lower()
        if text_lower in self.seen_quotes:
            return True
        
        # Near-duplicate check (fuzzy matching)
        for seen_text in self.seen_normalized:
            similarity = self._similarity(text_lower, seen_text)
            if similarity >= self.dedup_threshold:
                return True
        
        # Not a duplicate - add to seen sets
        self.seen_quotes.add(text_lower)
        self.seen_normalized.append(text_lower)
        
        # Limit memory usage - keep only last 10000 quotes for fuzzy matching
        if len(self.seen_normalized) > 10000:
            self.seen_normalized = self.seen_normalized[-10000:]
        
        return False
    
    def _similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        return SequenceMatcher(None, text1, text2).ratio()
    
    def reset_deduplication(self):
        """Reset the deduplication cache."""
        self.seen_quotes.clear()
        self.seen_normalized.clear()
        logger.info("Deduplication cache reset")


def main():
    """Test the text cleaner."""
    cleaner = TextCleaner()
    
    test_cases = [
        "This is a &quot;test&quot; with HTML entities&nbsp;and extra  spaces",
        "Check out https://example.com for more info [1]",
        "Unicode test: café, naïve, 日本語",
        '"Fancy quotes" and \'apostrophes\'',
        "Duplicate test",
        "Duplicate test",  # Should be filtered
        "Very similar duplicate test",  # Should be filtered (>95% similar)
    ]
    
    print("Testing TextCleaner:\n")
    for i, test in enumerate(test_cases, 1):
        result = cleaner.clean(test)
        print(f"{i}. Input:  {test}")
        print(f"   Output: {result}")
        print()


if __name__ == '__main__':
    main()
