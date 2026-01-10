"""
Configuration settings for the Wikiquote NLP System
"""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")  # Default database name
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")

# NLP Configuration
SPACY_MODEL = "en_core_web_sm"
LANGUAGE_FILTER = "en"
DEDUP_THRESHOLD = 0.95  # Similarity threshold for near-duplicate detection

# Retrieval Configuration
MAX_RESULTS = 10
FUZZY_EDIT_DISTANCE = 2
FULLTEXT_INDEX_NAME = "quoteTextIndex"

# Batch Processing
BATCH_SIZE = 1000  # Number of records to process in a single transaction

# XML Parsing
XML_FILE = DATA_DIR / "enwikiquote-20251220-pages-articles 2.xml"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
