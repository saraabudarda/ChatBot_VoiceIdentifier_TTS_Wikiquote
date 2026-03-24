"""
Author Mapper Module

Maps work names and character names to actual authors for accurate attribution.
Prevents showing "Hamlet" instead of "William Shakespeare" in responses.
"""

from typing import Optional, Tuple


# Work/Character name → Actual author mapping
WORK_TO_AUTHOR_MAP = {
    # William Shakespeare
    'hamlet': 'William Shakespeare',
    'macbeth': 'William Shakespeare',
    'romeo and juliet': 'William Shakespeare',
    'othello': 'William Shakespeare',
    'king lear': 'William Shakespeare',
    'the tempest': 'William Shakespeare',
    'a midsummer night\'s dream': 'William Shakespeare',
    'julius caesar': 'William Shakespeare',
    'merchant of venice': 'William Shakespeare',
    'twelfth night': 'William Shakespeare',
    
    # Other famous works
    'the prophet': 'Kahlil Gibran',
    'the bible': 'Various Authors (Biblical Scripture)',
    'the quran': 'Islamic Scripture',
    'the koran': 'Islamic Scripture',
    'the odyssey': 'Homer',
    'the iliad': 'Homer',
    'divine comedy': 'Dante Alighieri',
    'paradise lost': 'John Milton',
    'the republic': 'Plato',
    'meditations': 'Marcus Aurelius',
    'the art of war': 'Sun Tzu',
    'thus spoke zarathustra': 'Friedrich Nietzsche',
    'beyond good and evil': 'Friedrich Nietzsche',
    'the prince': 'Niccolò Machiavelli',
    'walden': 'Henry David Thoreau',
    'leaves of grass': 'Walt Whitman',
    'the waste land': 'T.S. Eliot',
    'pride and prejudice': 'Jane Austen',
    '1984': 'George Orwell',
    'animal farm': 'George Orwell',
    'brave new world': 'Aldous Huxley',
    'fahrenheit 451': 'Ray Bradbury',
    'to kill a mockingbird': 'Harper Lee',
    'the great gatsby': 'F. Scott Fitzgerald',
    'moby dick': 'Herman Melville',
    'the catcher in the rye': 'J.D. Salinger',
}


# Additional context for famous works (optional)
WORK_CONTEXT = {
    'hamlet': 'Act III, Scene I',
    'macbeth': 'Tragedy',
    'romeo and juliet': 'Tragedy',
    'the prophet': 'Philosophical Poetry',
    'the odyssey': 'Epic Poem',
    'the iliad': 'Epic Poem',
}


class AuthorMapper:
    """Maps work/character names to actual authors for accurate attribution."""
    
    def __init__(self):
        self.work_map = WORK_TO_AUTHOR_MAP
        self.context_map = WORK_CONTEXT
    
    def map_author(self, author: str, work: Optional[str] = None) -> Tuple[str, str]:
        """
        Map work/character name to actual author.
        
        Args:
            author: Author name from database (may be work/character name)
            work: Work title (optional)
            
        Returns:
            Tuple of (actual_author, confidence_level)
            confidence_level: 'high', 'medium', 'low'
        """
        author_lower = author.lower() if author else ''
        work_lower = work.lower() if work else ''
        
        # Check if author is actually a work name
        if author_lower in self.work_map:
            return self.work_map[author_lower], 'high'
        
        # Check if work name can be mapped
        if work_lower in self.work_map:
            return self.work_map[work_lower], 'high'
        
        # Check partial matches (e.g., "Hamlet" in "Hamlet, Prince of Denmark")
        for work_name, actual_author in self.work_map.items():
            if work_name in author_lower or work_name in work_lower:
                return actual_author, 'high'
        
        # No mapping found - return original author
        # Confidence depends on whether it looks like a person's name
        if author and any(indicator in author_lower for indicator in ['william', 'albert', 'mark', 'oscar', 'friedrich']):
            return author, 'high'
        elif author and author != 'Unknown':
            return author, 'medium'
        else:
            return author or 'Unknown', 'low'
    
    def get_source_info(self, author: str, work: Optional[str] = None) -> str:
        """
        Get formatted source information.
        
        Args:
            author: Author name
            work: Work title
            
        Returns:
            Formatted source string (e.g., "Hamlet, Act III, Scene I")
        """
        work_lower = work.lower() if work else ''
        
        # Get context if available
        context = self.context_map.get(work_lower, '')
        
        if work and work != 'Unknown':
            if context:
                return f"{work}, {context}"
            return work
        
        return "Source unknown"
    
    def is_famous_work(self, author: str, work: Optional[str] = None) -> bool:
        """
        Check if this is a famous literary work.
        
        Args:
            author: Author name
            work: Work title
            
        Returns:
            True if famous work, False otherwise
        """
        author_lower = author.lower() if author else ''
        work_lower = work.lower() if work else ''
        
        return author_lower in self.work_map or work_lower in self.work_map


# Singleton instance
_mapper = AuthorMapper()


def map_author(author: str, work: Optional[str] = None) -> Tuple[str, str]:
    """Convenience function to map author."""
    return _mapper.map_author(author, work)


def get_source_info(author: str, work: Optional[str] = None) -> str:
    """Convenience function to get source info."""
    return _mapper.get_source_info(author, work)


def is_famous_work(author: str, work: Optional[str] = None) -> bool:
    """Convenience function to check if famous work."""
    return _mapper.is_famous_work(author, work)
