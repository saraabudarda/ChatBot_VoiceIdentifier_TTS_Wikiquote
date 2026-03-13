# Wikiquote NLP System

A production-ready end-to-end NLP system for quote retrieval, autocompletion, and multi-user voice interaction using Neo4j graph database and Streamlit UI.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Neo4j Database (running on localhost:7687)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run data ingestion
python scripts/run_ingestion.py --limit 100

# Launch Streamlit UI
streamlit run src/ui/streamlit_app.py
```

## 🎯 Features

- **Quote Autocompletion**: Complete partial quotes using full-text search
- **Author Attribution**: Identify quote sources and authors
- **Semantic Search**: Find quotes by topic, author, or work
- **Interactive UI**: Streamlit dashboard with chat interface
- **Voice Interaction**: ASR, speaker identification, and personalized TTS (architecture designed)

## 📚 Documentation

For comprehensive documentation, please see the [`docs/`](docs/) folder:

- **[Complete Documentation](docs/README.md)** - Full project documentation, API reference, and usage examples
- **[Neo4j Setup Guide](docs/NEO4J_SETUP.md)** - Database installation and configuration
- **[Search Improvements](docs/IMPROVEMENTS.md)** - Enhanced ranking algorithm and performance metrics

## 📁 Project Structure

```
wiki db/
├── src/              # Source code
│   ├── ingestion/    # Data processing pipeline
│   ├── database/     # Neo4j integration
│   ├── retrieval/    # Quote search
│   ├── chatbot/      # Conversational AI
│   ├── voice/        # Voice interaction
│   └── ui/           # Streamlit interface
├── scripts/          # Utility scripts
├── tests/            # Unit tests
├── data/             # Data files
├── docs/             # Documentation
└── requirements.txt  # Dependencies
```

## 🔧 Configuration

Default Neo4j connection settings (edit `config.py` to customize):
```python
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_PASSWORD = "12345678"
```

## 📧 Support

For detailed setup instructions, troubleshooting, and advanced features, see the [complete documentation](docs/README.md).
