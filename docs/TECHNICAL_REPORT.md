# Technical Report: Multi-User Voice-Enabled Wikiquote NLP System

**Authors:** Sara Aboudarda, Shayan Ekramnia  
**Supervisor:** Prof. Francesco Cutugno  
**Institution:** Università degli Studi di Napoli Federico II  
**Course:** Natural Language Processing — May 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Data Engineering](#3-data-engineering)
4. [Core ML Modules](#4-core-ml-modules)
5. [Retrieval & Ranking](#5-retrieval--ranking)
6. [Implementation Details](#6-implementation-details)
7. [Evaluation & Results](#7-evaluation--results)
8. [Challenges & Solutions](#8-challenges--solutions)
9. [Limitations & Future Work](#9-limitations--future-work)
10. [Conclusion](#10-conclusion)
11. [References](#11-references)

---

## 1. Introduction

### 1.1 Motivation

Wikiquote is one of the largest open-source quote repositories, containing over **1.3 million quotations** from historical figures, literary works, films, and cultural sources. However, it provides no voice-based access, no speaker-aware personalization, and no conversational search interface.

Existing quote retrieval systems are text-only and lack the multimodal capabilities that modern NLP technologies can provide. This project addresses that gap by building a **complete end-to-end voice-enabled quote retrieval pipeline** that combines automatic speech recognition (ASR), speaker biometric identification, graph-based knowledge retrieval, and personalized text-to-speech synthesis.

### 1.2 Problem Statement

The project tackles four interconnected challenges:

1. **Voice-to-text for literary language** — Standard ASR models perform poorly on archaic and poetic vocabulary (e.g., *thee*, *wherefore*, *doth*) that is common in Wikiquote.
2. **Multi-user identification without login** — Traditional authentication requires manual credentials; voice biometrics offer a more natural alternative.
3. **Knowledge retrieval from noisy data** — The Wikiquote XML dump contains significant noise (movie dialogue, bibliographic citations, parody quotes) alongside genuine quotations.
4. **Personalized audio response** — Each identified user should receive TTS audio tailored to their stored voice preferences.

### 1.3 Research Questions

- **RQ1.** Can a domain-prompted Whisper ASR model accurately transcribe literary and archaic speech for downstream quote retrieval?
- **RQ2.** Can ECAPA-TDNN speaker embeddings reliably distinguish enrolled users from short (3–5 second) voice samples in a real-time web application?
- **RQ3.** Does the combined end-to-end pipeline deliver a satisfactory user experience within acceptable latency constraints on consumer-grade CPU hardware?

### 1.4 Objectives

| Objective | Description |
|-----------|-------------|
| **Data Engineering** | Parse the full English Wikiquote XML dump, clean and deduplicate 1.3M entries, model as a Neo4j knowledge graph |
| **ML Pipeline** | Integrate Whisper (ASR), ECAPA-TDNN (Speaker ID), and gTTS/Coqui (TTS) in a unified Python service |
| **UX & Interface** | Build a Streamlit web app with real-time voice recording, editable transcript review, ranked results, and personalized audio playback |

---

## 2. System Architecture

### 2.1 High-Level Design

The system follows a **two-phase pipeline architecture** with a human-in-the-loop review step bridging input understanding and retrieval:

```
Voice Input → Whisper ASR → ECAPA-TDNN Speaker ID → [User Reviews Transcript]
    → Intent Router → Neo4j Full-Text Search → Quality Ranking → TTS Output
```

**Phase 1 — Transcription & Speaker Identification:**
- Audio is captured via the browser's `MediaRecorder API` (wrapped by `streamlit-audiorecorder`)
- `faster-whisper` transcribes with a literary domain prompt
- SpeechBrain ECAPA-TDNN extracts a 192-dimensional x-vector and matches against enrolled profiles
- Both processes operate on the same audio file — no duplicate recording is required

**Phase 2 — Retrieval & Response:**
- The user reviews and optionally corrects the transcript in an editable text area
- A rule-based intent router classifies the query type (quote-text vs. author vs. topic)
- Neo4j full-text index (Lucene BM25) returns top-5 ranked quotes with similarity scores
- The best match is synthesized via gTTS using the identified speaker's voice preferences

### 2.2 Project Structure

The codebase comprises **6,338 lines of Python** across 24 source files organized into 7 modules:

```
ChatBot_VoiceIdentifier_TTS_Wikiquote/
├── src/
│   ├── chatbot/          # Intent routing, response generation, author mapping
│   │   ├── simple_router.py       # Regex-based intent classifier
│   │   ├── response_generator.py  # Anti-hallucination response builder
│   │   ├── intent_recognizer.py   # Query type detection
│   │   └── author_mapper.py       # Work-to-author resolution (40+ mappings)
│   ├── database/         # Neo4j client, schema, indexing
│   │   ├── neo4j_client.py        # Driver wrapper, connection pooling
│   │   ├── schema.py              # Node/relationship creation
│   │   └── indexing.py            # Full-text index management
│   ├── ingestion/        # XML parser, text cleaner, NLP processor
│   │   ├── xml_parser.py          # SAX-based Wikiquote XML parser
│   │   ├── text_cleaner.py        # 15+ regex quality filters
│   │   └── nlp_processor.py       # Linguistic preprocessing
│   ├── retrieval/        # Autocomplete engine, ranking
│   ├── speaker/          # Embedding extractor, identifier, profile manager
│   │   ├── embedding_extractor.py # ECAPA-TDNN wrapper
│   │   ├── identifier.py          # Cosine similarity matching
│   │   └── profile_manager.py     # Neo4j speaker CRUD
│   ├── ui/               # Streamlit dashboard (4 pages)
│   │   └── streamlit_app.py       # Main app (800+ lines)
│   └── voice/            # ASR, TTS, orchestrator
│       ├── asr_whisper.py         # faster-whisper integration (264 lines)
│       ├── tts_module.py          # gTTS + macOS fallback (300 lines)
│       ├── tts_coqui.py           # Coqui VITS multi-speaker (332 lines)
│       └── orchestrator.py        # Full pipeline coordination (301 lines)
├── models/               # Pre-trained model checkpoints
├── scripts/              # Data ingestion pipelines
├── config.py             # Centralized configuration
└── requirements.txt      # Dependencies
```

### 2.3 Module Responsibilities

| Module | Classes | Responsibility |
|--------|---------|---------------|
| `chatbot/` | `SimpleRouter`, `ResponseGenerator`, `AuthorMapper` | Intent classification, anti-hallucination response generation, work→author resolution |
| `database/` | `Neo4jClient` | Driver wrapper, schema creation, full-text index management, connection pooling |
| `ingestion/` | `XMLParser`, `TextCleaner`, `NLPProcessor` | SAX-based XML parsing, regex quality filtering, batch loading (1K records/batch) |
| `speaker/` | `VoiceEmbeddingExtractor`, `SpeakerIdentifier`, `SpeakerProfileManager` | ECAPA-TDNN embedding extraction, cosine similarity matching, Neo4j profile CRUD |
| `voice/` | `ASRWhisper`, `TTSEngine`, `CoquiTTS`, `VoiceOrchestrator` | Whisper transcription, multi-engine TTS synthesis, end-to-end pipeline coordination |
| `ui/` | Streamlit pages | Four-tab dashboard: Chatbot, Speaker ID, TTS, Voice Chat |

---

## 3. Data Engineering

### 3.1 Data Source

The dataset is derived from the **English Wikiquote XML dump (December 2025)**. Wikiquote pages are structured as MediaWiki XML, with each page representing a person, work, film, or topic. Quotes are embedded within wiki markup alongside editorial commentary, citations, and metadata.

### 3.2 Ingestion Pipeline

```
Wikiquote XML Dump → SAX Parser → Text Cleaner → Batch Loader → Neo4j Graph
```

1. **XML Parsing** — A custom SAX-based parser (`xml_parser.py`) streams the XML dump without loading it entirely into memory. It extracts page titles, section headings (used as work/source names), and quote text from within `<text>` elements.

2. **Text Cleaning** — `text_cleaner.py` applies **15+ regex filters** to exclude non-quote content:

   | Filter | Pattern | Purpose |
   |--------|---------|---------|
   | Min length | `len(text) >= 120` | Exclude fragments |
   | Page refs | `p\.\s*\d+` | Remove bibliographic citations |
   | URLs | `https?://` | Remove web links |
   | Brackets | `\[.*\]` | Remove stage directions |
   | Excessive punctuation | `!.*!.*!` | Remove screenplay dialogue |
   | Incomplete text | `&c.` | Remove truncated entries |
   | Year patterns | `\(\d{4}\)` | Remove date annotations |
   | Ellipses | `\.\.\.\.` | Remove incomplete quotes |

3. **Batch Loading** — Records are inserted into Neo4j in batches of 1,000 via the Bolt protocol, using `MERGE` statements to avoid duplicates at the node level.

4. **Deduplication** — Near-duplicate detection at cosine similarity threshold **0.95** removes entries that are trivially different (e.g., punctuation variants).

### 3.3 Neo4j Graph Schema

**Node Labels:**

| Label | Key Properties | Approximate Count |
|-------|---------------|-------------------|
| `Person` | `name`, `roles` | ~45,000 |
| `Quote` | `text`, `quality_score`, `is_canonical` | ~1,300,000 |
| `Work` | `name` | ~12,000 |
| `Source` | `name`, `type` | ~8,000 |

**Relationships:**

| Type | Direction | Semantics |
|------|-----------|-----------|
| `SAID` | Person → Quote | Direct attribution |
| `WROTE` | Person → Work | Authorship |
| `HAS_QUOTE` | Work → Quote | Containment |

**Indexes:**

Full-text Lucene indexes are created on `Quote.text`, `Person.name`, and `Work.name` for sub-100ms BM25 retrieval:

```cypher
CREATE FULLTEXT INDEX quoteIndex FOR (q:Quote) ON EACH [q.text];
CREATE FULLTEXT INDEX personIndex FOR (p:Person) ON EACH [p.name];
CREATE FULLTEXT INDEX workIndex FOR (w:Work) ON EACH [w.name];
```

### 3.4 Data Quality Analysis

Despite filtering, the corpus retains significant noise:

| Category | Estimated % | Example |
|----------|-------------|---------|
| Clean quotes | ~72% | *"The only thing we have to fear is fear itself."* |
| Movie/TV dialogue | ~11% | *"Code 10! Code 10!"* |
| Bibliographic citations | ~8% | *"Thomas Paine, in Life and Writings… p. 13"* |
| Screenplay/stage directions | ~5% | *"[picket sign] COGITO ERGO NOTHING!"* |
| Near-duplicates | ~4% | Punctuation and whitespace variants |

---

## 4. Core ML Modules

### 4.1 Automatic Speech Recognition — Whisper

**Model Selection:**

| Property | Value |
|----------|-------|
| Implementation | **faster-whisper** (CTranslate2-optimized) |
| Model size | `base` — 74M parameters |
| Speedup vs. reference | ~4× faster than PyTorch Whisper |
| Inference hardware | CPU (Apple Silicon) |
| Measured latency | **2–3 seconds** per utterance |
| Input format | 16kHz mono WAV |

**Domain Prompt Strategy:**

A custom `initial_prompt` is injected to bias Whisper's language model toward literary vocabulary:

> *"Quote search. The user may say a famous literary quote, an author name, a work title, or a topic. Literary and poetic language may appear."*

This reduces Word Error Rate on archaic terms like *thee*, *wherefore*, *doth*, and *hath* that are common in Wikiquote but rare in general speech corpora.

**Human-in-the-Loop Design:**

Rather than silently passing ASR output to search, the transcript is shown in an **editable text area** for user correction. This is a deliberate design decision: a single misheard word in a literary quote can completely change the search result.

### 4.2 Speaker Identification — ECAPA-TDNN

**Architecture:**

ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN) extends the conventional Time-Delay Neural Network with three innovations:

1. **Squeeze-Excitation (SE) blocks** — Channel-wise attention to emphasize speaker-discriminative frequency bands
2. **Multi-layer feature aggregation (MFA)** — Combines outputs from all TDNN layers, capturing both local and global temporal patterns
3. **Channel-dependent attention** — Attentive statistics pooling across the time axis

The SpeechBrain pre-trained model was trained on **VoxCeleb1 + VoxCeleb2** (7,000+ speakers, millions of utterances).

**Identification Pipeline:**

1. **Enrollment** — User records 3–5 second voice sample → 192-dim x-vector extracted → stored as Speaker node in Neo4j
2. **Extraction** — On each query, extract x-vector from the same audio used for ASR
3. **Matching** — Compute cosine similarity against all enrolled speaker embeddings
4. **Decision** — If max similarity ≥ **τ = 0.55**, return speaker name + confidence; else return "Guest"

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Embedding dimensions | 192 | Compact yet discriminative |
| Threshold τ | 0.55 | Balanced precision/recall |
| Min enrollment duration | 3 seconds | Sufficient for stable embedding |
| Measured accuracy | **81.5%** | On enrolled test speakers |

### 4.3 Text-to-Speech

**Engine Stack:**

| Engine | Type | Latency | Use Case |
|--------|------|---------|----------|
| **gTTS** | Cloud API (Google) | ~1–2s | Primary — best prosody |
| **Coqui VITS** | Local neural model | ~2–4s | Multi-speaker, offline |
| **macOS `say`** | OS built-in | <0.5s | Offline fallback |

**Per-Speaker Personalization:**

Each enrolled speaker has stored preferences:
- Voice gender (male / female / neutral)
- Speaking speed and regional accent
- Preferred TTS engine

Preferences are loaded automatically upon speaker identification. Guest users receive default settings.

**Audio-Text Validation:**

The orchestrator calls `_validate_audio_text()` before synthesis, performing an exact string match between the displayed quote and the TTS input. This prevents any mismatch between what is shown and what is spoken. Zero mismatches have been observed in testing.

---

## 5. Retrieval & Ranking

### 5.1 Intent Classification

A lightweight, deterministic regex router classifies each query with **zero latency** and full interpretability:

| Pattern | Detected Intent | Example |
|---------|----------------|---------|
| `^(quotes? )?by [A-Z]...` | Author search | "by Einstein" |
| `^what did [A-Z]... say` | Author search | "what did Gandhi say" |
| `^(find )?from [A-Z]...` | Work search | "from Hamlet" |
| (default) | Quote-text search | "to be or not to be" |

Author names are resolved via `difflib.SequenceMatcher` with fuzzy edit distance ≤ 2 (e.g., "Shakespear" → "William Shakespeare"). The `author_mapper.py` module contains 40+ famous work-to-author mappings for cases where the database stores work names instead of author names.

### 5.2 Neo4j Full-Text Search

```cypher
CALL db.index.fulltext.queryNodes('quoteIndex', $query)
YIELD node AS q, score
MATCH (p:Person)-[:SAID]->(q)
WHERE size(q.text) >= 120
  AND NOT q.text =~ '.*\\[.*\\].*'
  AND NOT q.text =~ '.*p\\.\\s*\\d+.*'
  AND NOT q.text =~ '.*https?://.*'
RETURN q.text, p.name, score * q.quality_score AS rank
ORDER BY rank DESC LIMIT 5
```

### 5.3 Quality Ranking

The ranking algorithm combines BM25 similarity with quality scoring:

- **Boost factors:** canonical quotes, famous authors (Shakespeare, Einstein, Gandhi), longer text
- **Penalty factors:** bibliographic citations, stage directions, duplicates, low `quality_score`
- **Final score** = BM25 × quality_score → normalized to percentage for display

---

## 6. Implementation Details

### 6.1 Streamlit Dashboard

The UI (`streamlit_app.py`, 800+ lines) provides four tabs:

| Tab | Functionality |
|-----|--------------|
| **🤖 Chatbot** | Text-based search with suggestion pills (Authors, Works, Moods, Topics), live autocomplete, conversational interface |
| **🎤 Speaker ID** | Voice enrollment (3–5 sec sample), identification test, profile management (view, delete) |
| **🔊 TTS** | Text input → speech synthesis, voice preference configuration, audio download |
| **🎙️ Voice Chat** | Full pipeline: Record → Transcribe → Identify → Review → Search → TTS playback |

**Key UX Decisions:**
- **Suggestion pills** provide one-click exploration without typing
- **Editable transcript** allows error correction before search
- **"Record Again" button** resets voice state without losing app session
- **Similarity percentages** on results help users gauge relevance

### 6.2 Voice Pipeline Orchestration

The `VoiceOrchestrator` class coordinates the full pipeline:

```python
class VoiceOrchestrator:
    def process_voice_query(self, audio_bytes):
        # 1. Write audio to temp file
        wav_path = self._save_temp_audio(audio_bytes)
        
        # 2. Parallel: ASR + Speaker ID
        transcript = self.asr.transcribe(wav_path, prompt=DOMAIN_PROMPT)
        speaker = self.speaker_id.identify(wav_path)
        
        # 3. User reviews transcript (in Streamlit UI)
        # 4. Intent routing
        intent = self.router.classify(transcript)
        
        # 5. Neo4j search
        results = self.search(intent, transcript)
        
        # 6. TTS with speaker preferences
        self._validate_audio_text(results[0].text)
        audio = self.tts.synthesize(results[0].text, speaker.preferences)
        
        return transcript, speaker, results, audio
```

### 6.3 Configuration

Centralized in `config.py`:

```python
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
WHISPER_MODEL = "base"
SPEAKER_THRESHOLD = 0.55
MIN_QUOTE_LENGTH = 120
BATCH_SIZE = 1000
```

---

## 7. Evaluation & Results

### 7.1 Performance Metrics

| Component | Metric | Value | Hardware |
|-----------|--------|-------|----------|
| ASR (Whisper base) | Latency | ~2–3s | CPU |
| Speaker ID (ECAPA) | Latency | ~0.5–1s | CPU |
| Speaker ID (ECAPA) | Accuracy | **81.5%** | CPU |
| Neo4j Full-Text | Latency | <100ms | Local |
| TTS (gTTS) | Latency | ~1–2s | Cloud API |
| TTS (Coqui VITS) | Latency | ~2–4s | CPU |
| Full pipeline | End-to-end | **~5–8s** | CPU total |

### 7.2 Database Statistics

| Metric | Value |
|--------|-------|
| Total quotes indexed | 1,300,000+ |
| Person nodes | ~45,000 |
| Work nodes | ~12,000 |
| Speaker embedding dims | 192 |
| Similarity threshold τ | 0.55 |
| Deduplication threshold | 0.95 cosine |
| Quality filters applied | 15+ regex patterns |

### 7.3 Working Features

- ✅ Multi-user speaker enrollment & identification
- ✅ Voice chat with full 6-stage pipeline (Record → Transcribe → Identify → Review → Search → TTS)
- ✅ Live autocomplete chatbot with suggestion pills
- ✅ Personalized TTS with audio-text validation
- ✅ Author attribution via graph traversal and fuzzy matching
- ✅ Editable transcript review (human-in-the-loop)
- ✅ "Record Again" state reset without page refresh
- ✅ Downloadable synthesized audio files

---

## 8. Challenges & Solutions

### 8.1 Data Quality Noise

**Problem:** The Wikiquote XML dump contains ~1.3M entries, but a significant portion consists of movie dialogue, screenplay fragments, bibliographic citations, and parody quotes.

**Examples of noisy data:**
- `"Code 10! Code 10!"` — movie dialogue
- `"[picket sign] COGITO ERGO NOTHING!"` — screenplay with stage directions
- `"Thomas Paine, in Life and Writings… p. 13"` — bibliographic citation

**Solution:** 15+ regex filters applied during ingestion and at query time. Minimum text length set to 120 characters. Quality scoring penalizes entries matching noise patterns.

**Status:** ⚠️ **Partially Resolved** — Filters exclude most metadata, but some noise passes through. An ML-based quote quality classifier (e.g., fine-tuned BERT binary classifier: real quote vs. noise) would be the ideal solution.

### 8.2 Missing Canonical Quotes

**Problem:** Descartes' *"Cogito ergo sum"*, several Socrates quotes, and other famous philosophical quotations are absent from the database. The database contains references *to* these figures but not quotes *by* them.

**Root Cause:** The Wikiquote XML parser encountered non-standard page formatting for these entries. This is a data-ingestion gap, not a retrieval failure.

**Status:** ❌ **Unresolved** — Requires re-parsing the XML dump with stricter quote boundary detection or supplementing with additional quote databases.

### 8.3 CPU Latency Bottleneck

**Problem:** Full pipeline takes 6–10 seconds on CPU. Primary bottlenecks: Whisper ASR (~3s) and Coqui TTS (~3s when used).

**Mitigations Applied:**
- `faster-whisper` CTranslate2 backend (4× speedup over reference PyTorch)
- Model caching via Streamlit `session_state` (avoids reloading on each interaction)
- gTTS cloud API as primary TTS (faster than local Coqui VITS)

**Projected:** GPU deployment (CUDA / Apple MPS) would reduce total latency to **<2 seconds**.

### 8.4 ASR Accuracy on Archaic Language

**Problem:** Standard Whisper models struggle with literary vocabulary. Example: *"wherefore art thou"* transcribed as *"where for art thou"*.

**Solutions:**
1. Literary domain prompt injected via `initial_prompt` parameter — biases the language model toward archaic vocabulary
2. Editable transcript review — users can correct ASR errors before search (human-in-the-loop)

### 8.5 Quote Attribution Accuracy

**Problem:** The database stores work names (e.g., "Hamlet") instead of author names ("William Shakespeare") in many cases.

**Solution:** `author_mapper.py` with 40+ famous work-to-author mappings with confidence levels (high/medium/low). All response methods use author mapping to resolve attributions.

### 8.6 Audio-Text Synchronization

**Problem:** Risk of TTS speaking different content than the displayed quote due to wrong index, race condition, or caching artifact.

**Solution:** `_validate_audio_text()` in the orchestrator asserts exact string match between the quote shown on screen and the text passed to the TTS engine. Any mismatch raises an exception and logs a detailed error.

**Status:** ✅ **Resolved** — Zero mismatches in testing.

### 8.7 Duplicate Results

**Problem:** Many quotes appear multiple times in search results due to duplicate entries in the database.

**Root Cause:** The Wikiquote XML dump contains the same quote on multiple pages (e.g., the author page and the work page).

**Status:** ⚠️ **Partially Addressed** — Quality scoring penalizes duplicates, but a `UNIQUE` constraint on `Quote.text` has not yet been implemented.

---

## 9. Limitations & Future Work

### 9.1 Data Improvements
- Re-parse Wikiquote XML with stricter quote boundary detection
- Train a supervised quote quality classifier (BERT-based binary: real quote vs. noise)
- Integrate supplementary quote databases (BrainyQuote, Goodreads Quotes)
- Add Neo4j `UNIQUE` constraint on `Quote.text` to prevent duplicate entries
- Implement user-driven feedback loop for quality rating

### 9.2 Model Upgrades
- Upgrade ASR to Whisper `medium` or `large-v3` for improved accuracy on archaic text
- Fine-tune ECAPA-TDNN on literary reading speech domain data
- Implement speaker adaptation — update embeddings with successive voice samples over time
- Explore voice cloning for truly personalized TTS synthesis
- GPU inference deployment (CUDA / Apple MPS) for <2s end-to-end latency

### 9.3 System Extensions
- Semantic search via sentence-transformer embeddings (beyond keyword BM25 matching)
- Multi-language support — French, German, Arabic Wikiquote editions
- REST API layer for third-party integration
- Progressive Web App (PWA) for mobile access
- Real-time streaming ASR (instead of record-then-transcribe batch mode)

---

## 10. Conclusion

This project demonstrates a **complete, end-to-end, multi-user voice-enabled NLP system** that bridges speech input, knowledge retrieval, and personalized audio output. The system integrates state-of-the-art ML models — Whisper for ASR, ECAPA-TDNN for speaker identification — with a Neo4j graph database containing 1.3M+ quotes and a real-time Streamlit web interface, all operating on consumer-grade CPU hardware.

The pipeline successfully processes voice queries through six stages (Record → Transcribe → Identify → Review → Search → Respond) with an end-to-end latency of ~5–8 seconds on CPU. Speaker identification achieves 81.5% accuracy on enrolled users with 3-second enrollment samples.

The primary remaining challenge is **data quality**: the Wikiquote corpus contains noise that regex filters cannot fully eliminate. Future work should prioritize ML-based quality classification, semantic search capabilities, and GPU deployment for sub-2-second latency.

---

## 11. References

1. Radford, A., Kim, J.W., et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision.* Proceedings of ICML.
2. Desplanques, B., Thienpondt, J., & Demuynck, K. (2020). *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification.* Proc. Interspeech.
3. Kim, J., Kong, J., & Son, J. (2021). *Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech.* Proceedings of ICML (VITS architecture).
4. Ravanelli, M., et al. (2021). *SpeechBrain: A General-Purpose Speech Toolkit.* arXiv:2106.04624.
5. Neo4j, Inc. (2024). *Neo4j Graph Database Documentation.* https://neo4j.com/docs/
6. Wikimedia Foundation. (2025). *Wikiquote Data Dumps.* https://dumps.wikimedia.org/

---

## Technical Stack Summary

| Layer | Technology | Version/Model |
|-------|-----------|---------------|
| Frontend | Streamlit + streamlit-audiorecorder | 1.x |
| Backend | Python | 3.10 |
| Database | Neo4j (graph database) | 5.x |
| ASR | faster-whisper (CTranslate2) | Whisper `base` (74M) |
| Speaker ID | SpeechBrain ECAPA-TDNN | 192-dim x-vectors |
| TTS (Primary) | gTTS (Google TTS API) | Cloud |
| TTS (Local) | Coqui TTS (VITS) | Multi-speaker |
| TTS (Fallback) | macOS `say` | OS built-in |
| Intent Routing | Custom regex + difflib | Rule-based |
| Search | Neo4j Full-Text (Lucene BM25) | Native index |
