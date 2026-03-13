# Wikiquote NLP System

A production-ready end-to-end NLP system for quote retrieval, autocompletion, and multi-user voice interaction using Neo4j graph database and Streamlit UI.

## 🎯 Features

### Core Functionality (Implemented)
- **Quote Autocompletion**: Complete partial quotes using full-text search
- **Author Attribution**: Identify quote sources and authors
- **Semantic Search**: Find quotes by topic, author, or work
- **Graph Database**: Neo4j-powered knowledge graph
- **NLP Pipeline**: Text cleaning, normalization, and linguistic processing
- **Interactive UI**: Streamlit dashboard with chat interface
- **Statistics**: Database analytics and top authors

### Voice Interaction (Architecture Designed)
- **ASR Module**: Speech-to-text using Whisper/NeMo
- **Speaker Identification**: Voice embeddings with Titanet
- **Personalized TTS**: Text-to-speech with voice customization
- **Voice Flow**: Complete voice interaction pipeline

## 📋 Requirements

### System Requirements
- Python 3.8+
- Neo4j Enterprise (running on localhost:7687)
- 8GB+ RAM recommended
- ~1GB disk space for models

### Python Dependencies
```bash
pip install -r requirements.txt
```

Core dependencies:
- `neo4j==5.15.0` - Graph database driver
- `streamlit==1.29.0` - Web UI framework
- `spacy==3.7.2` - NLP processing
- `lxml==5.1.0` - XML parsing
- `langdetect==1.0.9` - Language detection

## 🚀 Quick Start

### 1. Setup Neo4j

Ensure Neo4j is running:
```bash
# Default connection
URI: neo4j://127.0.0.1:7687
Database: neo4j
Password: 12345678
```

### 2. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 3. Run Data Ingestion

Process the Wikiquote XML dump and populate Neo4j:

```bash
# Full ingestion (takes 30-60 minutes)
python scripts/run_ingestion.py

# Test with limited pages
python scripts/run_ingestion.py --limit 100

# Clear database before ingestion
python scripts/run_ingestion.py --clear
```

### 4. Launch Streamlit UI

```bash
streamlit run src/ui/streamlit_app.py
```

Navigate to `http://localhost:8501`

## 📁 Project Structure

```
wiki db/
├── data/
│   └── enwikiquote-20251220-pages-articles 2.xml
├── src/
│   ├── ingestion/          # Data processing pipeline
│   │   ├── xml_parser.py   # Streaming XML parser
│   │   ├── text_cleaner.py # Text normalization
│   │   └── nlp_processor.py # spaCy processing
│   ├── database/           # Neo4j integration
│   │   ├── neo4j_client.py # Database client
│   │   ├── schema.py       # Graph schema
│   │   └── indexing.py     # Full-text indexes
│   ├── retrieval/          # Quote search
│   │   ├── autocomplete.py # Autocompletion engine
│   │   └── ranker.py       # Result ranking
│   ├── chatbot/            # Conversational AI
│   │   ├── intent_recognizer.py # Intent classification
│   │   └── response_generator.py # NLG responses
│   ├── voice/              # Voice interaction (design)
│   │   ├── asr_module.py   # Speech recognition
│   │   ├── speaker_id.py   # Speaker identification
│   │   └── tts_module.py   # Text-to-speech
│   └── ui/
│       └── streamlit_app.py # Web interface
├── scripts/
│   └── run_ingestion.py    # Data pipeline runner
├── tests/                  # Unit tests
├── config.py               # Configuration
├── requirements.txt        # Dependencies
└── README.md
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Neo4j
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_PASSWORD = "12345678"

# NLP
SPACY_MODEL = "en_core_web_sm"
DEDUP_THRESHOLD = 0.95

# Retrieval
MAX_RESULTS = 10
FUZZY_EDIT_DISTANCE = 2
```

## 💬 Usage Examples

### Streamlit UI

**Quote Completion:**
```
User: "To be or not to be"
Bot: Here's the complete quote:
     "To be, or not to be, that is the question."
     — William Shakespeare, from Hamlet
```

**Author Search:**
```
User: "Quotes by Einstein"
Bot: Here are some quotes by Albert Einstein:
     1. "Imagination is more important than knowledge."
     ...
```

**Attribution:**
```
User: "Who said 'I think therefore I am'?"
Bot: That quote is by René Descartes, from Discourse on the Method.
```

### Python API

```python
from src.database.neo4j_client import Neo4jClient
from src.retrieval.autocomplete import QuoteAutocomplete
import config

# Connect to database
client = Neo4jClient(
    uri=config.NEO4J_URI,
    user=config.NEO4J_USER,
    password=config.NEO4J_PASSWORD
)

# Search for quotes
autocomplete = QuoteAutocomplete(client)
results = autocomplete.complete_quote("to be or not")

for quote in results:
    print(f"{quote['quote']} — {quote['author']}")
```

## 🧪 Testing

Run tests:
```bash
# All tests
pytest tests/

# Specific test
pytest tests/test_parser.py -v

# With coverage
pytest --cov=src tests/
```

## 📊 Database Schema

### Nodes
- **Quote**: `id`, `text_raw`, `text_normalized`, `language`
- **Author**: `name`, `birth_year`
- **Work**: `title`, `type`
- **Category**: `name`

### Relationships
- `(Author)-[:SAID]->(Quote)`
- `(Quote)-[:FROM_WORK]->(Work)`
- `(Quote)-[:IN_CATEGORY]->(Category)`

### Indexes
- Full-text index on `Quote.text_raw` and `Quote.text_normalized`
- B-tree indexes on author names and work titles

## 🎤 Voice Interaction (Future)

The voice modules are architecturally designed but require additional setup:

### ASR (Speech Recognition)
```bash
pip install openai-whisper
```

```python
from src.voice.asr_module import ASRModule

asr = ASRModule(model_type='whisper', model_size='base')
result = asr.transcribe('audio.wav')
print(result['text'])
```

### Speaker Identification
```bash
pip install nemo-toolkit[asr]
```

```python
from src.voice.speaker_id import SpeakerIdentification

speaker_id = SpeakerIdentification(model_name='titanet')
speaker_id.register_speaker('user1', 'enrollment.wav')
result = speaker_id.identify_speaker('query.wav')
print(f"Speaker: {result['speaker_id']}")
```

### TTS (Text-to-Speech)
```bash
pip install TTS
```

```python
from src.voice.tts_module import TTSModule

tts = TTSModule(model_type='coqui')
tts.set_voice_style('user1', {'speaker': 'female'})
audio_path = tts.synthesize("Hello world", speaker_id='user1')
```

## 🔍 Troubleshooting

### Neo4j Connection Failed
- Ensure Neo4j is running: `neo4j status`
- Check credentials in `config.py`
- Verify port 7687 is accessible

### Out of Memory During Ingestion
- Reduce batch size: `--batch-size 500`
- Process in chunks: `--limit 10000`
- Increase system RAM or use swap

### spaCy Model Not Found
```bash
python -m spacy download en_core_web_sm
```

### Full-Text Index Missing
```python
from src.database.indexing import IndexManager
manager = IndexManager(client)
manager.create_fulltext_index()
```

## 📈 Performance

- **Ingestion**: ~30-60 minutes for full Wikiquote dump
- **Query Speed**: <100ms for autocomplete
- **Database Size**: ~2-3GB for complete dataset
- **Memory Usage**: ~2-4GB during ingestion

## 🤝 Contributing

This is an academic NLP project. Key principles:
- **Modular**: Each component is self-contained
- **Documented**: Comprehensive docstrings
- **Tested**: Unit and integration tests
- **Production-ready**: Error handling and logging

## 📝 License

Academic project for NLP research.

## 🙏 Acknowledgments

- **Wikiquote**: Quote data source
- **Neo4j**: Graph database platform
- **spaCy**: NLP processing
- **Streamlit**: UI framework
- **OpenAI Whisper**: ASR model
- **NVIDIA NeMo**: Speaker identification and TTS

## 📧 Support

For issues or questions, refer to:
- Neo4j documentation: https://neo4j.com/docs/
- spaCy documentation: https://spacy.io/
- Streamlit documentation: https://docs.streamlit.io/
