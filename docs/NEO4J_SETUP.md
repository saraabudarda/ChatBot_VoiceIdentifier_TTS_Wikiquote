# Neo4j Setup Guide

## Problem Identified
Neo4j is not currently running or installed on your system.

## Solution Options

### Option 1: Install Neo4j Desktop (Recommended for Development)

1. **Download Neo4j Desktop:**
   - Visit: https://neo4j.com/download/
   - Download Neo4j Desktop for macOS
   - Install the application

2. **Create a Database:**
   - Open Neo4j Desktop
   - Click "New" → "Create Project"
   - Click "Add" → "Local DBMS"
   - Name: `wikiquote`
   - Password: `12345678` (or update `config.py` with your password)
   - Version: Latest (5.x recommended)
   - Click "Create"

3. **Start the Database:**
   - Click "Start" on your database
   - Wait for it to show "Active"
   - Default connection: `neo4j://localhost:7687`

### Option 2: Install Neo4j via Homebrew

```bash
# Install Neo4j
brew install neo4j

# Start Neo4j service
brew services start neo4j

# Or start manually
neo4j start

# Check status
neo4j status
```

**Set Password:**
```bash
# First time setup - set password
neo4j-admin set-initial-password 12345678
```

### Option 3: Use Docker (Quick Start)

```bash
# Pull and run Neo4j container
docker run \
    --name neo4j-wikiquote \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/12345678 \
    -d neo4j:latest

# Check if running
docker ps | grep neo4j
```

## Verify Connection

After starting Neo4j, test the connection:

```bash
cd "/Users/sara/Desktop/wiki db"
python3 -c "from src.database.neo4j_client import Neo4jClient; import config; client = Neo4jClient(uri=config.NEO4J_URI, user=config.NEO4J_USER, password=config.NEO4J_PASSWORD, database=config.NEO4J_DATABASE); print('✓ Connected!' if client.test_connection() else '✗ Failed')"
```

## Access Neo4j Browser

Once running, access the web interface:
- URL: http://localhost:7474
- Username: `neo4j`
- Password: `12345678` (or your chosen password)

## Troubleshooting

### Port Already in Use
```bash
# Check what's using port 7687
lsof -i :7687

# Kill the process if needed
kill -9 <PID>
```

### Wrong Password
If you set a different password, update `config.py`:
```python
NEO4J_PASSWORD = "your_password_here"
```

Or set environment variable:
```bash
export NEO4J_PASSWORD="your_password_here"
```

### Connection Refused
- Ensure Neo4j is running: `neo4j status` or check Docker
- Check firewall settings
- Verify port 7687 is not blocked

## Next Steps

1. Choose one of the installation options above
2. Start Neo4j
3. Verify connection using the test command
4. Run the data ingestion: `python scripts/run_ingestion.py --limit 100`
5. Launch Streamlit: `streamlit run src/ui/streamlit_app.py`
