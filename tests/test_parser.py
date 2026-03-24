"""
Unit tests for XML Parser
"""
import pytest
from src.ingestion.xml_parser import WikiquoteParser
from pathlib import Path


class TestWikiquoteParser:
    """Test cases for WikiquoteParser class."""
    
    def test_clean_wiki_markup(self):
        """Test wiki markup cleaning."""
        parser = WikiquoteParser(Path("dummy.xml"))
        
        # Test wiki links
        text = "[[Albert Einstein|Einstein]] was a physicist"
        cleaned = parser._clean_wiki_markup(text)
        assert "Einstein was a physicist" in cleaned
        
        # Test bold/italic
        text = "'''bold''' and ''italic''"
        cleaned = parser._clean_wiki_markup(text)
        assert "bold and italic" in cleaned
    
    def test_split_sections(self):
        """Test section splitting."""
        parser = WikiquoteParser(Path("dummy.xml"))
        
        text = """
== Introduction ==
Some intro text

== Quotes ==
* Quote 1
* Quote 2

== External Links ==
Links here
"""
        
        sections = parser._split_sections(text)
        assert len(sections) >= 3
        
        section_titles = [title for title, _ in sections]
        assert "Quotes" in section_titles
    
    def test_extract_author(self):
        """Test author extraction."""
        parser = WikiquoteParser(Path("dummy.xml"))
        
        # Normal author page
        author = parser._extract_author("Albert Einstein", "")
        assert author == "Albert Einstein"
        
        # Special page (should return Unknown)
        author = parser._extract_author("Category:Science", "")
        assert author == "Unknown"
    
    def test_find_quotes(self):
        """Test quote finding."""
        parser = WikiquoteParser(Path("dummy.xml"))
        
        text = """
* This is a quote
* Another quote here
* Short
"""
        
        quotes = parser._find_quotes(text)
        
        # Should find quotes but filter very short ones
        assert len(quotes) >= 1
        assert "This is a quote" in quotes[0] or "Another quote here" in quotes[0]


def test_parser_initialization():
    """Test parser can be initialized."""
    xml_file = Path("test.xml")
    parser = WikiquoteParser(xml_file)
    
    assert parser.xml_file == xml_file
    assert parser.namespace is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
