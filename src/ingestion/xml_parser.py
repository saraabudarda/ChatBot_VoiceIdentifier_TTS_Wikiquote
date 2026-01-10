"""
Enhanced XML Parser with Role Taxonomy and Entity Type Detection

This parser extracts quotes and classifies entities as Person, Work, or Source
with role taxonomy mapping.
"""
import xml.etree.ElementTree as ET
import json
import re
import logging
from typing import Iterator, Dict, Optional, Set
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Role Taxonomy Map
ROLE_MAP = {
    # Scientists
    "physicists": "Scientist", "mathematicians": "Scientist", "biologists": "Scientist", 
    "chemists": "Scientist", "astronomers": "Scientist", "scientists": "Scientist",
    # Artists/Authors
    "writers": "Author", "novelists": "Author", "poets": "Author", "playwrights": "Author",
    "painters": "Artist", "sculptors": "Artist", "architects": "Artist",
    # Performers
    "actors": "Actor", "actresses": "Actor", "directors": "Director", "musicians": "Musician", 
    "singers": "Musician", "composers": "Musician",
    # Leaders
    "politicians": "Politician", "presidents": "Politician", "monarchs": "Politician", 
    "prime ministers": "Politician", "revolutionaries": "Activists",
    # Thinkers
    "philosophers": "Philosopher", "theologians": "Philosopher"
}


class EnhancedWikiquoteParser:
    """
    Enhanced parser with entity type detection and role taxonomy.
    
    Classifies entities as:
    - Person: Individual people with roles (Scientist, Author, etc.)
    - Work: Books, films, plays, speeches
    - Source: Generic sources
    """
    
    def __init__(self, xml_file: Path):
        """
        Initialize the parser.
        
        Args:
            xml_file: Path to Wikiquote XML dump
        """
        self.xml_file = xml_file
        self.namespace = 'http://www.mediawiki.org/xml/export-0.11/'
    
    def parse(self, limit: Optional[int] = None) -> Iterator[Dict]:
        """
        Parse XML dump and yield quote data.
        
        Args:
            limit: Maximum number of pages to process
            
        Yields:
            Dictionary with quote data including entity type and roles
        """
        logger.info(f"Starting to parse {self.xml_file}")
        
        count = 0
        quote_count = 0
        
        try:
            context = ET.iterparse(str(self.xml_file), events=('end',))
            
            for event, elem in context:
                if elem.tag == f'{{{self.namespace}}}page':
                    try:
                        # Extract page data
                        title_elem = elem.find(f'{{{self.namespace}}}title')
                        if title_elem is None:
                            continue
                        title = title_elem.text
                        
                        # Skip non-main namespace
                        ns_elem = elem.find(f'{{{self.namespace}}}ns')
                        if ns_elem is None or ns_elem.text != '0':
                            elem.clear()
                            continue
                        
                        # Get page text
                        revision = elem.find(f'{{{self.namespace}}}revision')
                        if revision is None:
                            continue
                        text_elem = revision.find(f'{{{self.namespace}}}text')
                        if text_elem is None or not text_elem.text:
                            elem.clear()
                            continue
                        
                        text_content = text_elem.text
                        
                        # Analyze page
                        entity_data = self._analyze_page(title, text_content)
                        
                        # Extract quotes
                        for quote_text in self._extract_quotes(text_content):
                            quote_data = {
                                'quote_raw': quote_text,
                                'author': title,
                                'real_author': entity_data['real_author'],
                                'entity_type': entity_data['type'],
                                'roles': entity_data['roles'],
                                'categories': entity_data['categories'],
                                'work': entity_data.get('work', 'Unknown'),
                                'section': 'Quotes'
                            }
                            
                            yield quote_data
                            quote_count += 1
                        
                        count += 1
                        if limit and count >= limit:
                            break
                    
                    except Exception as e:
                        logger.debug(f"Error processing page: {e}")
                    finally:
                        elem.clear()
        
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
        
        logger.info(f"Parsing complete. Processed {count} pages, extracted {quote_count} quotes")
    
    def _analyze_page(self, title: str, text_content: str) -> Dict:
        """
        Analyze page to determine entity type and roles.
        
        Args:
            title: Page title
            text_content: Page wiki text
            
        Returns:
            Dictionary with entity analysis
        """
        categories = re.findall(r'\[\[Category:(.*?)\]\]', text_content)
        
        entity_type = "Source"  # Default
        roles = set()
        real_author = None
        
        for cat in categories:
            cat_lower = cat.lower()
            
            # Extract Real Author from Work Categories
            work_match = re.search(
                r'(?:films|novels|plays|books|speeches|essays) by (.+)', 
                cat, 
                re.IGNORECASE
            )
            if work_match:
                entity_type = "Work"
                # Clean author name (remove sort keys)
                raw_auth = work_match.group(1).split('|')[0]
                real_author = raw_auth.strip()
            
            # Role Mapping
            for keyword, role in ROLE_MAP.items():
                if keyword in cat_lower:
                    roles.add(role)
                    if entity_type == "Source":
                        entity_type = "Person"
            
            # Generic Person Check
            if any(x in cat_lower for x in ["births", "deaths", "living people"]):
                if entity_type == "Source":
                    entity_type = "Person"
        
        return {
            'type': entity_type,
            'roles': list(roles),
            'real_author': real_author,
            'categories': categories
        }
    
    def _extract_quotes(self, text_content: str) -> Iterator[str]:
        """
        Extract quotes from page text.
        
        Args:
            text_content: Wiki markup text
            
        Yields:
            Quote text strings
        """
        lines = text_content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Match bulleted quotes, quoted text, OR dialogue (colon-prefixed)
            if line.startswith('*') or line.startswith('""') or line.startswith('"') or line.startswith(':'):
                # Basic cleanup
                quote_text = line.lstrip('*: ').strip()
                
                # Skip short or invalid quotes
                if len(quote_text) < 20:
                    continue
                
                # Skip metadata
                if any(quote_text.startswith(x) for x in [
                    '[[Category:', '{{', 'File:', 'Image:', '==',
                    'See also', 'External links', 'References'
                ]):
                    continue
                
                # Clean wiki markup
                quote_text = self._clean_wiki_markup(quote_text)
                
                if len(quote_text) >= 20:
                    yield quote_text
    
    def _clean_wiki_markup(self, text: str) -> str:
        """
        Remove wiki markup from text.
        
        Args:
            text: Text with wiki markup
            
        Returns:
            Cleaned text
        """
        # Remove bold/italic
        text = re.sub(r"'{2,5}", '', text)
        
        # Replace wiki links [[target|display]] or [[target]]
        def replace_link(match):
            parts = match.group(1).split('|')
            return parts[-1] if len(parts) > 1 else parts[0]
        
        text = re.sub(r'\[\[([^\]]+)\]\]', replace_link, text)
        
        # Remove templates
        text = re.sub(r'\{\{[^}]+\}\}', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


def main():
    """Test the parser."""
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python xml_parser.py <xml_file>")
        sys.exit(1)
    
    xml_file = Path(sys.argv[1])
    parser = EnhancedWikiquoteParser(xml_file)
    
    print("Testing parser on first 5 pages...\n")
    for i, quote_data in enumerate(parser.parse(limit=5)):
        if i < 10:
            print(f"{i+1}. [{quote_data['entity_type']}] {quote_data['author']}")
            print(f"   Roles: {quote_data['roles']}")
            print(f"   Quote: {quote_data['quote_raw'][:80]}...")
            print()


if __name__ == '__main__':
    main()
