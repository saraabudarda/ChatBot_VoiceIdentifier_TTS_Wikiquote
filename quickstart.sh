#!/bin/bash

# Quick Start Script for Wikiquote NLP System
# This script helps you get started with the system

echo "🚀 Wikiquote NLP System - Quick Start"
echo "======================================"
echo ""

# Check Neo4j connection
echo "1️⃣  Testing Neo4j connection..."
python3 -c "
from src.database.neo4j_client import Neo4jClient
import config
try:
    client = Neo4jClient(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD, config.NEO4J_DATABASE)
    if client.test_connection():
        print('   ✅ Neo4j connected successfully!')
        client.close()
    else:
        print('   ❌ Neo4j connection failed')
        exit(1)
except Exception as e:
    print(f'   ❌ Error: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "Please ensure Neo4j is running and configured correctly."
    exit 1
fi

echo ""
echo "2️⃣  Choose an option:"
echo ""
echo "   A) Test ingestion (100 pages, ~2 minutes)"
echo "   B) Full ingestion (~30-60 minutes)"
echo "   C) Skip ingestion and launch UI"
echo ""
read -p "Enter your choice (A/B/C): " choice

case $choice in
    [Aa]* )
        echo ""
        echo "Running test ingestion (100 pages)..."
        python3 scripts/run_ingestion.py --limit 100
        ;;
    [Bb]* )
        echo ""
        echo "Running full ingestion (this will take 30-60 minutes)..."
        python3 scripts/run_ingestion.py
        ;;
    [Cc]* )
        echo ""
        echo "Skipping ingestion..."
        ;;
    * )
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "3️⃣  Launching Streamlit UI..."
echo ""
echo "   The application will open in your browser at:"
echo "   http://localhost:8501"
echo ""
echo "   Press Ctrl+C to stop the server"
echo ""

streamlit run src/ui/streamlit_app.py
