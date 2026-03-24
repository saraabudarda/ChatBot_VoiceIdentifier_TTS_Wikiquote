# Technical Report: Multi-User Voice-Enabled Wikiquote Chatbot

## Project Overview

A multi-user voice-enabled chatbot system that retrieves quotes from Wikiquote, identifies speakers, and responds with personalized text-to-speech audio.

## System Architecture

### Core Components

1. **Database**: Neo4j graph database storing 1.3M+ quotes with relationships
2. **Frontend**: Streamlit web application with voice recording capabilities
3. **Backend**: Python-based microservices architecture

### Models & Technologies Used

| Component | Technology | Model/Version |
|-----------|-----------|---------------|
| **Speech Recognition (ASR)** | faster-whisper | Whisper Tiny |
| **Speaker Identification** | SpeechBrain | ECAPA-TDNN embeddings |
| **Text-to-Speech (TTS)** | Coqui TTS | VITS multi-speaker |
| **Database** | Neo4j | 5.x with full-text indexing |
| **Intent Recognition** | Rule-based NLP | Custom pattern matching |
| **Embedding Storage** | NumPy arrays | 192-dimensional vectors |

### Voice Pipeline

```
Audio Input → ASR (Whisper) → Speaker ID (ECAPA-TDNN) → Intent Recognition 
→ Database Query (Neo4j) → Response Generation → TTS (Coqui) → Audio Output
```

## Implementation Details

### 1. Quote Retrieval System

- **Full-text search** using Neo4j indexes
- **Ranking algorithm** prioritizing famous authors (Shakespeare, Einstein)
- **Author mapping** to resolve work names (Hamlet → William Shakespeare)
- **Quality filtering** to exclude metadata and citations

### 2. Speaker Identification

- **Voice embeddings** extracted using SpeechBrain ECAPA-TDNN
- **Cosine similarity** matching against enrolled profiles
- **Threshold-based** identification (75% confidence minimum)
- **Personalization** via stored TTS preferences (voice gender, style)

### 3. Text-to-Speech

- **Multi-speaker synthesis** using Coqui TTS
- **Voice preferences** stored per user (male/female/neutral)
- **Audio validation** ensuring TTS matches displayed quote exactly

## Challenges Encountered & Solutions

### Challenge 1: Quote Attribution Accuracy
**Problem**: Database stored work names (e.g., "Hamlet") instead of authors ("William Shakespeare")

**Solution**: 
- Created `author_mapper.py` with 40+ famous work-to-author mappings
- Implemented confidence levels (high/medium/low) for uncertain attributions
- Updated all response methods to use author mapping

### Challenge 2: Data Quality Issues
**Problem**: Database contained metadata, citations, and movie dialogue instead of real quotes

**Examples**:
- "Thomas Paine, in Life and Writings of Thomas Paine. p. 13" (citation)
- "Code 10! Code 10!" (movie dialogue)
- "[picket sign] COGITO ERGO NOTHING!" (screenplay with stage directions)

**Solutions Implemented**:
```cypher
-- Quality filters in Neo4j queries
WHERE size(q.text) >= 120                    -- Minimum length
  AND NOT q.text =~ '.*\\(\\d{4}\\).*'       -- No years
  AND NOT q.text =~ '.*p\\. \\d+.*'          -- No page numbers
  AND NOT q.text =~ '.*https?://.*'          -- No URLs
  AND NOT q.text =~ '.*\\[.*\\].*'           -- No brackets (stage directions)
  AND NOT q.text =~ '.*\\.\\.\\.\\.*'        -- No ellipses
  AND NOT q.text =~ '.*!.*!.*!.*'            -- No multiple exclamations
  AND NOT q.text CONTAINS '&c.'              -- No incomplete text
```

**Status**: Partially resolved - filters exclude most metadata, but some movie dialogue still passes through

### Challenge 3: Audio-Text Synchronization
**Problem**: Risk of TTS speaking different content than displayed quote

**Solution**:
- Implemented `_validate_audio_text()` method in orchestrator
- Validates exact match between displayed quote and TTS input
- Logs mismatches and raises errors if validation fails

### Challenge 4: Missing Famous Quotes
**Problem**: Database lacks actual quotes from famous philosophers (e.g., Descartes' "Cogito ergo sum")

**Status**: **UNRESOLVED** - Data limitation, not code issue. Database contains references TO Descartes but not quotes BY Descartes.

## Current Challenges

### 1. Data Quality (Ongoing)
- **Issue**: Database has ~1.3M quotes but includes significant noise:
  - Movie/TV dialogue and screenplay fragments
  - Citations and bibliographic references
  - Parody quotes and joke variations
  
- **Impact**: Users searching for famous quotes often get low-quality results
  
- **Current Mitigation**: 
  - 15+ regex filters to exclude known patterns
  - Minimum length requirement (120 characters)
  - Ordering by quote length (longer = more substantive)
  
- **Limitation**: Cannot filter all bad quotes without excluding some good ones

### 2. Missing Canonical Quotes
- **Issue**: Database missing actual quotes from famous figures (Descartes, Socrates, etc.)
- **Root Cause**: Wikiquote XML parsing may have missed certain quote formats
- **Impact**: Searches for famous philosophical quotes return parodies instead

### 3. Duplicate Results
- **Issue**: Many quotes appear 2x in results
- **Root Cause**: Database contains duplicate entries
- **Status**: Not yet addressed

## Performance Metrics

| Metric | Value |
|--------|-------|
| Database Size | 1.3M quotes |
| ASR Latency | ~2-3 seconds (Whisper Tiny) |
| Speaker ID Accuracy | 81.5% (in testing) |
| TTS Generation | ~1-2 seconds per quote |
| End-to-End Latency | ~5-8 seconds |

## System Capabilities

✅ **Working Features**:
- Multi-user speaker identification
- Personalized TTS with voice preferences
- Quote search with intent recognition
- Author attribution with confidence levels
- Audio validation (TTS matches display)
- Live search with autocomplete
- Voice chat with full pipeline

⚠️ **Partial/Limited**:
- Quote quality (filtered but not perfect)
- Coverage of famous quotes (data gaps)

## Recommendations

1. **Data Cleanup**: Re-parse Wikiquote XML with stricter quote extraction rules
2. **Deduplication**: Add database constraint to prevent duplicate quotes
3. **Quality Scoring**: Implement ML-based quote quality classifier
4. **Fallback Sources**: Integrate additional quote databases for missing content
5. **User Feedback**: Add quote rating system to identify low-quality results

## Technical Stack Summary

```
Frontend:  Streamlit + streamlit-audiorecorder
Backend:   Python 3.x
Database:  Neo4j 5.x (graph database)
ASR:       faster-whisper (Whisper Tiny)
Speaker:   SpeechBrain (ECAPA-TDNN)
TTS:       Coqui TTS (VITS)
NLP:       Custom intent recognition
```

## Conclusion

The system successfully implements a complete voice-enabled quote retrieval pipeline with speaker identification and personalized TTS. The main technical challenges have been addressed through author mapping, quality filtering, and audio validation. The primary remaining challenge is **data quality** - the underlying Wikiquote database contains significant noise that cannot be fully filtered programmatically without also excluding legitimate quotes. A data re-ingestion or supplementary data source would be the most effective solution.
