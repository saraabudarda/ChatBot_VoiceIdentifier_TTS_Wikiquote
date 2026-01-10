"""
NLP Processor for Linguistic Normalization

This module applies linguistic processing using spaCy to create
normalized versions of quotes while preserving the original text.
"""
import logging
from typing import Dict, List, Optional
import spacy
from spacy.tokens import Doc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLPProcessor:
    """
    Linguistic normalization processor using spaCy.
    
    Creates normalized versions of quotes through tokenization,
    lemmatization, and stopword handling while preserving sentence
    boundaries and the original text.
    """
    
    def __init__(self, model_name: str = 'en_core_web_sm'):
        """
        Initialize the NLP processor.
        
        Args:
            model_name: spaCy model to use for processing
        """
        self.model_name = model_name
        self.nlp = None
        self._load_model()
    
    def _load_model(self):
        """Load the spaCy model."""
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except OSError:
            logger.warning(f"Model {self.model_name} not found. Attempting to download...")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', self.model_name])
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Downloaded and loaded spaCy model: {self.model_name}")
    
    def process(self, quote_data: Dict) -> Dict:
        """
        Process a quote record with NLP normalization.
        
        Args:
            quote_data: Dictionary with 'quote_raw' and other metadata
            
        Returns:
            Enhanced dictionary with 'quote_normalized' and NLP metadata
        """
        if not quote_data.get('quote_raw'):
            return quote_data
        
        raw_text = quote_data['quote_raw']
        
        # Process with spaCy
        doc = self.nlp(raw_text)
        
        # Create normalized version
        normalized_text = self._normalize_text(doc)
        
        # Extract linguistic features
        tokens = self._extract_tokens(doc)
        lemmas = self._extract_lemmas(doc)
        sentences = self._extract_sentences(doc)
        
        # Add to quote data
        quote_data['quote_normalized'] = normalized_text
        quote_data['tokens'] = tokens
        quote_data['lemmas'] = lemmas
        quote_data['sentence_count'] = len(sentences)
        quote_data['sentences'] = sentences
        
        return quote_data
    
    def _normalize_text(self, doc: Doc) -> str:
        """
        Create normalized version of text.
        
        Normalization includes:
        - Lowercasing
        - Lemmatization
        - Stopword removal (but preserved in metadata)
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            Normalized text string
        """
        normalized_tokens = []
        
        for token in doc:
            # Skip punctuation and whitespace
            if token.is_punct or token.is_space:
                continue
            
            # Use lemma and lowercase
            lemma = token.lemma_.lower()
            
            # Keep all tokens (including stopwords) for better search
            # Stopword information is preserved in metadata
            normalized_tokens.append(lemma)
        
        return ' '.join(normalized_tokens)
    
    def _extract_tokens(self, doc: Doc) -> List[str]:
        """
        Extract tokens from document.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            List of token strings
        """
        return [token.text for token in doc if not token.is_space]
    
    def _extract_lemmas(self, doc: Doc) -> List[str]:
        """
        Extract lemmas from document.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            List of lemma strings
        """
        return [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    
    def _extract_sentences(self, doc: Doc) -> List[str]:
        """
        Extract sentences preserving boundaries.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            List of sentence strings
        """
        return [sent.text.strip() for sent in doc.sents]
    
    def batch_process(self, quote_records: List[Dict], batch_size: int = 100) -> List[Dict]:
        """
        Process multiple quotes in batches for efficiency.
        
        Args:
            quote_records: List of quote dictionaries
            batch_size: Number of quotes to process at once
            
        Returns:
            List of processed quote dictionaries
        """
        processed_records = []
        
        for i in range(0, len(quote_records), batch_size):
            batch = quote_records[i:i + batch_size]
            
            # Extract raw texts
            texts = [record.get('quote_raw', '') for record in batch]
            
            # Process batch with spaCy pipe (more efficient)
            docs = list(self.nlp.pipe(texts))
            
            # Process each doc
            for record, doc in zip(batch, docs):
                if doc.text:
                    # Create normalized version
                    normalized_text = self._normalize_text(doc)
                    tokens = self._extract_tokens(doc)
                    lemmas = self._extract_lemmas(doc)
                    sentences = self._extract_sentences(doc)
                    
                    # Add to record
                    record['quote_normalized'] = normalized_text
                    record['tokens'] = tokens
                    record['lemmas'] = lemmas
                    record['sentence_count'] = len(sentences)
                    record['sentences'] = sentences
                
                processed_records.append(record)
            
            if (i + batch_size) % 1000 == 0:
                logger.info(f"Processed {i + batch_size} quotes with NLP")
        
        return processed_records


def main():
    """Test the NLP processor."""
    processor = NLPProcessor()
    
    test_quote = {
        'quote_raw': 'To be, or not to be, that is the question. Whether it is nobler in the mind to suffer.',
        'author': 'William Shakespeare',
        'work': 'Hamlet',
        'section': 'Act III',
        'language': 'en'
    }
    
    print("Testing NLPProcessor:\n")
    print(f"Original: {test_quote['quote_raw']}\n")
    
    processed = processor.process(test_quote)
    
    print(f"Normalized: {processed['quote_normalized']}")
    print(f"Tokens: {processed['tokens']}")
    print(f"Lemmas: {processed['lemmas']}")
    print(f"Sentences: {processed['sentences']}")
    print(f"Sentence count: {processed['sentence_count']}")


if __name__ == '__main__':
    main()
