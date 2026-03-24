"""
Streamlit Dashboard for Wikiquote NLP System

Features:
- Live search autocomplete in chatbot
- Navigation for all features
- Speaker identification and TTS pages
- Statistics dashboard
"""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from src.database.neo4j_client import Neo4jClient
from src.retrieval.autocomplete import QuoteAutocomplete
from src.chatbot.simple_router import SimpleRouter, RouteType
from src.chatbot.response_generator import ResponseGenerator


# Page configuration
st.set_page_config(
    page_title="Wikiquote NLP System",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .quote-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .author-text {
        font-style: italic;
        color: #555;
        margin-top: 0.5rem;
    }
    .stat-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #666;
    }
    .search-result {
        padding: 0.8rem;
        margin: 0.3rem 0;
        background-color: #f8f9fa;
        border-radius: 4px;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .search-result:hover {
        background-color: #e9ecef;
    }
    .result-quote {
        font-size: 0.95rem;
        color: #1a1a1a;
    }
    .result-author {
        font-size: 0.85rem;
        color: #6c757d;
        font-style: italic;
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_database_client():
    """Initialize and cache the database client."""
    try:
        client = Neo4jClient(
            uri=config.NEO4J_URI,
            user=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD,
            database=config.NEO4J_DATABASE
        )
        return client
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        return None


@st.cache_resource
def get_autocomplete_engine(_client):
    """Initialize and cache the autocomplete engine."""
    return QuoteAutocomplete(_client)


@st.cache_resource
def get_intent_recognizer():
    """Initialize and cache the intent recognizer."""
    return IntentRecognizer()


@st.cache_resource
def get_simple_router():
    """Initialize and cache the simple router."""
    return SimpleRouter()


@st.cache_resource
def get_response_generator():
    """Initialize and cache the response generator."""
    return ResponseGenerator()


def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'db_connected' not in st.session_state:
        st.session_state.db_connected = False
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'show_autocomplete' not in st.session_state:
        st.session_state.show_autocomplete = False


def sidebar():
    """Render the sidebar navigation."""
    with st.sidebar:
        st.markdown("# 📚 Wikiquote NLP")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🤖 Chatbot & Search", "🎤 Speaker Identification", "🔊 Text-to-Speech", "🎙️ Voice Chat"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Connection status
        client = get_database_client()
        if client and client.test_connection():
            st.success("✓ Database Connected")
            st.session_state.db_connected = True
        else:
            st.error("✗ Database Disconnected")
            st.session_state.db_connected = False
        
        st.markdown("---")
        
        # Info
        st.markdown("### About")
        st.markdown("""
        This system provides:
        - Quote autocompletion
        - Author attribution
        - Semantic search
        - Multi-user voice interaction ✨
        """)
        
        return page


def chatbot_page():
    """Render the chatbot and search page with live autocomplete."""
    st.markdown('<div class="main-header">💬 Quote Chatbot & Search</div>', unsafe_allow_html=True)
    
    if not st.session_state.db_connected:
        st.error("⚠️ Database not connected. Please check your Neo4j connection.")
        return
    
    # Get components
    client = get_database_client()
    autocomplete = get_autocomplete_engine(client)
    router = get_simple_router()
    response_generator = get_response_generator()
    
    # --- Smart Suggestions ---
    st.markdown("### ✨ Smart Suggestions")
    st.caption("Click any suggestion to instantly search top-tier quotes.")
    
    import random
    
    # Pre-defined high-quality suggestion pools
    SUGGESTION_POOLS = {
        "👤 Authors": ["Albert Einstein", "Oscar Wilde", "Mahatma Gandhi", "Mark Twain", "William Shakespeare", "Jane Austen", "Friedrich Nietzsche"],
        "📚 Works": ["Hamlet", "The Odyssey", "Republic", "Frankenstein", "Leaves of Grass", "The Prophet"],
        "🎭 Moods": ["Courage", "Loneliness", "Friendship", "Love and loss", "Inspirational", "Hope", "Sorrow"],
        "💭 Topics": ["Philosophy", "Science", "Art", "Life and Death", "Time", "Justice", "Nature"]
    }
    
    # Randomly select a few to display on initial load, cached to prevent jumping on every keystroke
    if 'active_suggestions' not in st.session_state:
        st.session_state.active_suggestions = {
            cat: random.sample(items, 2) for cat, items in SUGGESTION_POOLS.items()
        }
        
    def apply_suggestion(text):
        """Callback to populate the search bar and trigger a rerun search."""
        st.session_state.live_search = text
        # Force the engine to recognize it as a new search even if clicked twice
        st.session_state.last_search = "" 
        
    cols = st.columns(4)
    for i, (category, items) in enumerate(st.session_state.active_suggestions.items()):
        with cols[i]:
            st.markdown(f"**{category}**")
            for item in items:
                # Format the internal search query logically based on category
                if category == "👤 Authors":
                    search_term = f"by {item}"
                elif category == "📚 Works":
                    search_term = f"from {item}"
                else:
                    search_term = item
                    
                st.button(
                    item, 
                    key=f"sugg_{category}_{item}", 
                    on_click=apply_suggestion, 
                    args=(search_term,),
                    use_container_width=True
                )
    
    st.markdown("---")
    
    # Live Search Section with AJAX-style updates
    st.markdown("### 🔍 Live Search")
    
    # Initialize search state
    if 'last_search' not in st.session_state:
        st.session_state.last_search = ""
    
    search_query = st.text_input(
        "Search for quotes...",
        placeholder="The night is darkening round me",
        key="live_search",
        on_change=lambda: setattr(st.session_state, 'search_changed', True)
    )
    
    # AJAX-style live update - only fetch if query changed
    if search_query and len(search_query) >= 3:
        if search_query != st.session_state.last_search:
            # Update results
            with st.spinner(""):
                st.session_state.search_results = autocomplete.complete_quote(search_query, max_results=5)
                st.session_state.last_search = search_query
        
        # Display results
        if st.session_state.search_results:
            st.markdown("**Search Results:**")
            
            # Use container for dynamic updates
            results_container = st.container()
            
            with results_container:
                for i, result in enumerate(st.session_state.search_results):
                    quote_text = result.get('quote', '')
                    author = result.get('author', 'Unknown')
                    
                    # Truncate for display
                    display_quote = quote_text if len(quote_text) <= 150 else quote_text[:147] + "..."
                    
                    # Create clickable result with better styling
                    with st.container():
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            if st.button(
                                f"📝 {display_quote}",
                                key=f"live_result_{i}",
                                help=f"Click to add to chat",
                                use_container_width=True
                            ):
                                # Add to chat
                                st.session_state.messages.append({
                                    "role": "user",
                                    "content": search_query
                                })
                                response = f'I found this quote:\n\n"{quote_text}"\n\n— {author}'
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": response
                                })
                                # Clear search
                                st.session_state.last_search = ""
                                st.session_state.search_results = []
                                st.rerun()
                        with col2:
                            st.caption(f"*{author}*")
                        st.divider()
        else:
            st.info("No results found. Try different keywords.")
    elif search_query and len(search_query) < 3:
        st.caption("Type at least 3 characters to search...")
    else:
        # Clear results when search is empty
        if st.session_state.last_search:
            st.session_state.search_results = []
            st.session_state.last_search = ""

    
    st.markdown("---")
    
    # Chat interface
    st.markdown("### 💬 Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask me about quotes..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                # Use simple router
                route_type, clarification, metadata = router.route(prompt)
                
                # Handle clarification
                if route_type == RouteType.CLARIFICATION:
                    response = clarification
                    st.markdown(response)
                else:
                    # Route to autocomplete (default for all queries)
                    results = autocomplete.complete_quote(prompt, max_results=5)
                    
                    # Generate response with anti-hallucination rules
                    response = response_generator.generate(None, results, metadata)
                    st.markdown(response)
                
                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": response})


def speaker_id_page():
    """Render the speaker identification page."""
    st.markdown('<div class="main-header">🎤 Speaker Identification</div>', unsafe_allow_html=True)
    
    if not st.session_state.db_connected:
        st.error("⚠️ Database not connected. Please check your Neo4j connection.")
        return
    
    # Get components
    client = get_database_client()
    
    # Initialize speaker components (lazy loading)
    if 'speaker_manager' not in st.session_state:
        try:
            with st.spinner("Loading speaker identification models..."):
                from src.speaker import SpeakerProfileManager, VoiceEmbeddingExtractor, SpeakerIdentifier
                
                st.session_state.speaker_manager = SpeakerProfileManager(client)
                st.session_state.embedding_extractor = VoiceEmbeddingExtractor()
                st.session_state.speaker_identifier = SpeakerIdentifier(
                    st.session_state.speaker_manager,
                    st.session_state.embedding_extractor,
                    threshold=0.55
                )
                
                st.success("✅ Speaker identification models loaded!")
                
        except Exception as e:
            st.error(f"❌ Failed to initialize speaker components: {e}")
            st.info("💡 Install dependencies: `pip install speechbrain librosa soundfile`")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
            return
    
    # Verify all components are initialized
    if 'speaker_identifier' not in st.session_state:
        st.error("❌ Speaker identifier not initialized. Please refresh the page.") 
        return
    
    # Tabs for different functions
    tab1, tab2, tab3 = st.tabs(["👤 Enroll Speaker", "🔍 Identify Speaker", "📋 Manage Profiles"])
    
    with tab1:
        st.markdown("### Enroll New Speaker")
        st.markdown("Record a voice sample to create a new speaker profile.")
        
        # Speaker name input
        speaker_name = st.text_input("Speaker Name", placeholder="Enter speaker name...")
        
        # Audio recording
        st.markdown("**Record Voice Sample** (3-5 seconds)")
        
        try:
            from audiorecorder import audiorecorder
            audio_bytes = audiorecorder(
                start_prompt="🎤 Start Recording",
                stop_prompt="⏹️ Stop Recording",
                pause_prompt="⏸️ Pause",
                key="enroll_recorder"
            )
            
            if audio_bytes and speaker_name:
                if st.button("✅ Create Profile", type="primary"):
                    with st.spinner("Processing voice sample..."):
                        try:
                            # Convert AudioSegment to numpy array
                            import io
                            import soundfile as sf
                            
                            # Export AudioSegment to WAV bytes
                            wav_io = io.BytesIO()
                            audio_bytes.export(wav_io, format="wav")
                            wav_io.seek(0)
                            
                            # Read with soundfile
                            audio_data, sample_rate = sf.read(wav_io)
                            
                            # Extract embedding
                            embedding = st.session_state.embedding_extractor.extract_from_array(
                                audio_data,
                                sample_rate
                            )
                            
                            # Create speaker profile
                            import uuid
                            speaker_id = f"speaker_{uuid.uuid4().hex[:8]}"
                            
                            st.session_state.speaker_manager.create_speaker(
                                speaker_id=speaker_id,
                                name=speaker_name,
                                embedding=embedding
                            )
                            
                            st.success(f"✅ Speaker profile created for {speaker_name}!")
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"Failed to create profile: {e}")
                            import traceback
                            st.code(traceback.format_exc())
            
            elif audio_bytes and not speaker_name:
                st.warning("Please enter a speaker name")
                
        except ImportError:
            st.warning("Audio recorder not available. Install with: `pip install streamlit-audiorecorder`")
            st.info("Alternative: Upload an audio file")
            
            uploaded_file = st.file_uploader("Upload voice sample", type=['wav', 'mp3', 'ogg'])
            
            if uploaded_file and speaker_name:
                if st.button("✅ Create Profile from File", type="primary"):
                    with st.spinner("Processing..."):
                        try:
                            # Save temporarily
                            import tempfile
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                                tmp.write(uploaded_file.read())
                                tmp_path = tmp.name
                            
                            # Extract embedding
                            embedding = st.session_state.embedding_extractor.extract_from_file(tmp_path)
                            
                            # Create profile
                            import uuid
                            speaker_id = f"speaker_{uuid.uuid4().hex[:8]}"
                            
                            st.session_state.speaker_manager.create_speaker(
                                speaker_id=speaker_id,
                                name=speaker_name,
                                embedding=embedding
                            )
                            
                            st.success(f"✅ Profile created for {speaker_name}!")
                            
                            # Cleanup
                            import os
                            os.unlink(tmp_path)
                            
                        except Exception as e:
                            st.error(f"Failed: {e}")
    
    with tab2:
        st.markdown("### Identify Speaker")
        st.markdown("Record audio to identify the speaker.")
        
        try:
            from audiorecorder import audiorecorder
            audio_bytes = audiorecorder(
                start_prompt="🎤 Start Recording",
                stop_prompt="⏹️ Stop Recording",
                pause_prompt="⏸️ Pause",
                key="identify_recorder"
            )
            
            if audio_bytes:
                if st.button("🔍 Identify", type="primary"):
                    with st.spinner("Identifying speaker..."):
                        try:
                            # Convert AudioSegment to numpy
                            import io
                            import soundfile as sf
                            
                            # Ensure speaker_identifier is initialized
                            if 'speaker_identifier' not in st.session_state:
                                st.error("❌ Speaker identifier not initialized. Please refresh the page.")
                                st.stop()
                            
                            # Export AudioSegment to WAV bytes
                            wav_io = io.BytesIO()
                            audio_bytes.export(wav_io, format="wav")
                            wav_io.seek(0)
                            
                            # Read with soundfile
                            audio_data, sample_rate = sf.read(wav_io)
                            
                            # Identify
                            speaker_id, confidence, speaker_data = st.session_state.speaker_identifier.identify_from_array(
                                audio_data,
                                sample_rate
                            )
                            
                            if speaker_id:
                                st.success(f"✅ Identified: **{speaker_data['name']}**")
                                st.metric("Confidence", f"{confidence:.1%}")
                            else:
                                st.warning(f"❌ Unknown speaker (best match: {confidence:.1%})")
                                
                        except Exception as e:
                            st.error(f"Identification failed: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                            
        except ImportError:
            st.info("Install audio recorder: `pip install streamlit-audiorecorder`")
    
    with tab3:
        st.markdown("### Speaker Profiles")
        
        # Get all speakers
        speakers = st.session_state.speaker_manager.get_all_speakers()
        
        if speakers:
            st.markdown(f"**Total Speakers:** {len(speakers)}")
            
            for speaker in speakers:
                with st.expander(f"👤 {speaker['name']}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Created:** {speaker.get('created_at', 'N/A')}")
                        st.write(f"**Last Seen:** {speaker.get('last_seen', 'N/A')}")
                        st.write(f"**Samples:** {speaker.get('sample_count', 1)}")
                    
                    with col2:
                        if st.button("🗑️ Delete", key=f"del_{speaker['speaker_id']}"):
                            st.session_state.speaker_manager.delete_speaker(speaker['speaker_id'])
                            st.success("Deleted!")
                            st.rerun()
        else:
            st.info("No speaker profiles yet. Enroll a speaker in the 'Enroll Speaker' tab.")


def voice_chat_page():
    """Render the voice chat page with full voice interaction."""
    st.markdown('<div class="main-header">🎙️ Voice Chat</div>', unsafe_allow_html=True)

    if not st.session_state.db_connected:
        st.error("⚠️ Database not connected. Please check your Neo4j connection.")
        return

    st.markdown("""
    Speak your query and get a personalized audio response! The system will:
    1. 🎤 Transcribe your speech
    2. 👤 Identify who you are
    3. 🔍 Search for relevant quotes
    4. 🔊 Respond with **your personalized voice**
    """)

    # ── Load ASR + Speaker ID (lazy, cached in session_state) ─────────────
    if 'vc_asr' not in st.session_state:
        with st.spinner("Loading voice models (first time only)…"):
            try:
                from src.voice.asr_whisper import ASRWhisper
                from src.speaker.profile_manager import SpeakerProfileManager
                from src.speaker.embedding_extractor import VoiceEmbeddingExtractor
                from src.speaker.identifier import SpeakerIdentifier

                client = get_database_client()
                st.session_state.vc_asr = ASRWhisper(model_size='base')
                spk_mgr = SpeakerProfileManager(client)
                emb_ext = VoiceEmbeddingExtractor()
                st.session_state.vc_speaker_id = SpeakerIdentifier(spk_mgr, emb_ext, threshold=0.55)
                st.session_state.vc_speaker_mgr = spk_mgr
                st.success("✅ Voice models ready!")
            except Exception as e:
                st.error(f"Failed to initialize voice components: {e}")
                st.info("Make sure all dependencies are installed: `pip install faster-whisper speechbrain`")
                return

    st.markdown("---")


    # ── Helper: reset all voice state ────────────────────────────────────
    def reset_voice_state():
        """Clear all voice-pipeline session state without touching app-wide state."""
        for _k in [
            'vc_audio_bytes', 'vc_transcript', 'vc_speaker_id_result',
            'vc_speaker_name', 'vc_speaker_confidence',
            'vc_search_results', 'vc_search_mode', 'vc_search_done',
            'vc_transcription_done',
        ]:
            st.session_state.pop(_k, None)

    # ── Phase 1: Audio input (shown only before transcription) ────────────
    if not st.session_state.get('vc_transcription_done'):

        audio_input_bytes = None

        st.markdown("**🎙️ Voice Recorder**")
        try:
            from audiorecorder import audiorecorder
            recorded = audiorecorder(
                start_prompt="⏺ Record",
                stop_prompt="⏹ Stop Recording",
                key="vc_recorder",
                show_visualizer=True,
            )
            if recorded and len(recorded) > 0:
                import io as _io
                wav_io = _io.BytesIO()
                recorded.export(wav_io, format="wav")
                audio_input_bytes = wav_io.getvalue()

                try:
                    import soundfile as sf
                    wav_io.seek(0)
                    audio_arr, sr = sf.read(wav_io)
                    if audio_arr.ndim > 1:
                        audio_arr = audio_arr.mean(axis=1)
                    step = max(1, len(audio_arr) // 600)
                    st.line_chart(audio_arr[::step], height=80, use_container_width=True)
                except Exception:
                    pass

                st.audio(audio_input_bytes, format="audio/wav")
        except ImportError:
            st.info("Install: `pip install streamlit-audiorecorder`")

        st.markdown("**📂 Or upload an audio file:**")
        uploaded = st.file_uploader(
            "Upload audio file",
            type=['wav', 'mp3', 'ogg', 'm4a'],
            key="vc_upload",
            label_visibility="collapsed",
        )
        if uploaded:
            audio_input_bytes = uploaded.read()
            st.audio(audio_input_bytes, format="audio/wav")

        # ── Transcribe + Speaker ID on button click ───────────────────────
        if audio_input_bytes and st.button("🚀 Find Quote", type="primary", use_container_width=True):
            import tempfile, os

            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(audio_input_bytes)
                tmp_path = tmp.name

            try:
                _DOMAIN_PROMPT = (
                    "Quote search. The user may say a famous literary quote, "
                    "an author name, a work title, or a topic. "
                    "Literary and poetic language may appear."
                )
                with st.spinner("🎤 Transcribing…"):
                    asr_result = st.session_state.vc_asr.transcribe_file(
                        tmp_path,
                        initial_prompt=_DOMAIN_PROMPT,
                    )
                    transcript = asr_result.get('text', '').strip()

                if not transcript:
                    st.warning("Could not transcribe audio. Please try again.")
                else:
                    with st.spinner("👤 Identifying speaker…"):
                        spk_id, spk_conf, spk_data = \
                            st.session_state.vc_speaker_id.identify_from_file(tmp_path)

                    st.session_state.vc_transcript         = transcript
                    st.session_state.vc_speaker_id_result  = spk_id
                    st.session_state.vc_speaker_name       = spk_data.get('name', 'Guest') if spk_id else 'Guest'
                    st.session_state.vc_speaker_confidence = spk_conf
                    st.session_state.vc_transcription_done = True
                    st.session_state.vc_search_done        = False
                    st.rerun()

            except Exception as e:
                st.error(f"Transcription failed: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # ── Phase 2: Transcript review + Search ──────────────────────────────
    else:
        st.markdown("---")
        st.markdown("### 📊 Results")

        # Editable transcript
        st.markdown("#### 1️⃣ Transcription — Review & Correct")
        transcript = st.text_area(
            "Whisper heard this. Fix any errors before searching:",
            value=st.session_state.get('vc_transcript', ''),
            height=80,
            key="vc_transcript_edit",
        )

        # Speaker recognition display
        st.markdown("#### 2️⃣ Speaker Recognition")
        spk_id   = st.session_state.get('vc_speaker_id_result')
        spk_name = st.session_state.get('vc_speaker_name', 'Guest')
        spk_conf = st.session_state.get('vc_speaker_confidence', 0.0)

        if spk_id:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success(f"👤 **Identified:** {spk_name}")
            with col2:
                st.metric("Confidence", f"{spk_conf:.1%}")
        else:
            st.warning(
                f"Unknown speaker (best: {spk_conf:.1%}). "
                "Enroll in **Speaker Identification** for personalized responses."
            )

        # Action buttons
        col_s, col_r = st.columns([2, 1])
        with col_s:
            do_search = st.button(
                "🔍 Search with this text",
                type="primary",
                use_container_width=True,
                disabled=not transcript.strip(),
            )
        with col_r:
            if st.button("🔄 Record Again", use_container_width=True):
                reset_voice_state()
                st.rerun()

        # ── Search (runs on click; result cached to avoid re-running) ─────
        if do_search or st.session_state.get('vc_search_done'):

            if not st.session_state.get('vc_search_done'):
                import re as _re
                from src.retrieval.autocomplete import QuoteAutocomplete
                client = get_database_client()
                autocomplete = QuoteAutocomplete(client)

                # Quote-text-first: only switch to author search if the
                # query STARTS with a strong, explicit author-request phrase.
                _STRONG_AUTHOR_PATS = [
                    r'^\s*(?:quotes?\s+)?by\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})',
                    r'^\s*(?:find\s+(?:me\s+)?a\s+quote\s+)?from\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})',
                    r'^\s*(?:what\s+did\s+)([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\s+say',
                    r'^\s*show\s+me\s+quotes?\s+(?:by|from)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})',
                ]
                _SKIP = {'the', 'a', 'an', 'this', 'that', 'my', 'your', 'me'}

                detected_author = None
                for pat in _STRONG_AUTHOR_PATS:
                    m = _re.search(pat, transcript, _re.IGNORECASE)
                    if m:
                        candidate = m.group(1).strip()
                        if candidate.lower() not in _SKIP and len(candidate) > 2:
                            detected_author = candidate
                            break

                with st.spinner("🔍 Searching quotes…"):
                    if detected_author:
                        results = autocomplete.find_by_author(detected_author, limit=5)
                        search_mode = f"Author search: *{detected_author}*"
                        if not results:
                            results = autocomplete.complete_quote(transcript, max_results=5)
                            search_mode = "Quote-text search (author not found, fell back)"
                    else:
                        results = autocomplete.complete_quote(transcript, max_results=5)
                        search_mode = "Quote-text search"

                st.session_state.vc_search_results = results
                st.session_state.vc_search_mode    = search_mode
                st.session_state.vc_search_done    = True

            results     = st.session_state.get('vc_search_results', [])
            search_mode = st.session_state.get('vc_search_mode', '')

            st.markdown("#### 3️⃣ Top Matching Quotes")
            st.caption(
                f"🔎 {search_mode} &nbsp;|&nbsp; 🎙️ Voice input &nbsp;|&nbsp; "
                f"Searched: *\"{transcript[:60]}{'…' if len(transcript) > 60 else ''}\"*"
            )

            if not results:
                st.warning("No matching quote found. Try editing the transcript above and search again.")
            else:
                top5      = results[:5]
                top_score = max((r.get('final_score', 1) for r in top5), default=1) or 1

                st.markdown("**🔍 Top Matches:**")
                for rank, result in enumerate(top5, start=1):
                    q_text    = result.get('quote', '')
                    q_author  = result.get('author', 'Unknown')
                    q_work    = result.get('work', '')
                    raw_score = result.get('final_score', 0)
                    pct       = int(round((raw_score / top_score) * 100))

                    try:
                        from src.chatbot.author_mapper import map_author
                        q_author, _ = map_author(q_author, q_work)
                    except Exception:
                        pass

                    display_text = q_text if len(q_text) <= 120 else q_text[:117] + "…"
                    author_part  = f" — {q_author}" + (f", {q_work}" if q_work and q_work != 'Unknown' else "")
                    is_top       = rank == 1
                    border_color = "#1f77b4" if is_top else "#adb5bd"
                    pct_color    = "#1f77b4" if is_top else "#6c757d"
                    weight       = "700" if is_top else "500"

                    st.markdown(
                        f"<div style='background:#f8f9fa; border-radius:8px; padding:10px 16px;"
                        f"border-left:4px solid {border_color}; margin-bottom:8px; line-height:1.5;'>"
                        f"<span style='font-size:0.82rem;font-weight:600;color:#6c757d;'>{rank}.&nbsp;</span>"
                        f"<span style='font-size:0.82rem;font-weight:700;color:{pct_color};'>[{pct}%]&nbsp;</span>"
                        f"<span style='font-size:0.93rem;font-weight:{weight};color:#1a1a1a;'>\u201c{display_text}\u201d</span>"
                        f"<span style='font-size:0.85rem;color:#555;font-style:italic;'>{author_part}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                # TTS: top result
                top        = results[0]
                quote_text = top.get('quote', '')
                author     = top.get('author', 'Unknown')
                work       = top.get('work', '')
                try:
                    from src.chatbot.author_mapper import map_author
                    actual_author, _ = map_author(author, work)
                except Exception:
                    actual_author = author

                voice_prefs = {'voice_gender': 'female', 'accent': 'auto', 'speed': 1.0}
                if spk_id:
                    try:
                        saved = st.session_state.vc_speaker_mgr.get_voice_preferences(spk_id)
                        if saved:
                            voice_prefs = saved
                    except Exception:
                        pass

                st.markdown("#### 4️⃣ Personalized Audio Response")
                gender = voice_prefs.get('voice_gender', 'female')
                accent = voice_prefs.get('accent', 'auto')
                speed  = float(voice_prefs.get('speed', 1.0))

                with st.spinner(f"🔊 Generating audio ({gender} voice)…"):
                    audio_bytes_out, audio_fmt = _generate_preview_audio(
                        text=quote_text,
                        voice_gender=gender,
                        accent=accent,
                        speed=speed,
                    )

                st.audio(audio_bytes_out, format=audio_fmt)

                if spk_id:
                    st.caption(f"🎙️ Voice personalized for **{spk_name}** ({gender}, speed {speed:.1f}x)")
                else:
                    st.caption("💡 Log in via Speaker Identification to get your own personalized voice.")

                ext = 'wav' if 'wav' in audio_fmt else 'mp3'
                st.download_button(
                    label="⬇️ Download Audio",
                    data=audio_bytes_out,
                    file_name=f"quote_response.{ext}",
                    mime=audio_fmt,
                )

                with st.expander("🔍 Debug Info"):
                    st.json({
                        'transcript_searched': transcript,
                        'route': search_mode,
                        'input_source': 'voice',
                        'speaker_id': spk_id,
                        'speaker_name': spk_name,
                        'speaker_confidence': f"{spk_conf:.3f}",
                        'voice_preferences': voice_prefs,
                        'top_quote_length': len(quote_text),
                    })

    # ── Example queries ───────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("💡 Example Voice Queries"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Quote Completion:**")
            st.code("To be or not to be")
            st.code("I think therefore")
            st.code("Tell me a quote about courage")
        with col2:
            st.markdown("**Author Queries:**")
            st.code("Show me quotes by Einstein")
            st.code("What did Shakespeare say?")
            st.code("Give me a random quote")
    




# ---------------------------------------------------------------------------
# TTS preview helper — generates distinct audio per voice type
# ---------------------------------------------------------------------------
def _generate_preview_audio(text: str, voice_gender: str, accent: str, speed: float):
    """
    Generate preview audio bytes using gTTS (female/neutral/default)
    or pyttsx3 (male system voice).

    Returns:
        Tuple of (audio_bytes: bytes, format_str: str) for st.audio()
    """
    import io, tempfile, os

    if voice_gender == 'male':
        # ── macOS `say` command — subprocess-based, safe inside Streamlit ───
        try:
            import subprocess, time

            with tempfile.NamedTemporaryFile(suffix='.aiff', delete=False) as f:
                tmp_path = f.name

            # `say` voices: Daniel (British), Alex (US), Fred (US), Tom (US)
            # Use Daniel as default male voice; adjust rate (words/min, default 175)
            rate = int(175 * speed)
            subprocess.run(
                ['say', '-v', 'Daniel', '-r', str(rate), '-o', tmp_path, text],
                check=True, capture_output=True, timeout=30
            )

            # Convert AIFF → WAV bytes with soundfile
            import soundfile as sf
            audio_data, sr = sf.read(tmp_path)
            wav_buf = io.BytesIO()
            sf.write(wav_buf, audio_data, sr, format='WAV')
            wav_buf.seek(0)
            os.unlink(tmp_path)
            return wav_buf.read(), 'audio/wav'

        except Exception as e:
            # Fall through to gTTS Canadian accent as male fallback
            import logging
            logging.getLogger(__name__).warning(f"macOS say failed, using gTTS: {e}")

    # ── gTTS: female / neutral / default ────────────────────────────────────
    from gtts import gTTS

    # Map gender → TLD accent for audible difference
    tld_defaults = {
        'female':  'com',        # US female
        'neutral': 'co.uk',      # British neutral
        'default': 'com.au',     # Australian
        'male':    'ca',         # Canadian (gTTS fallback for male)
    }
    tld = accent if accent != 'auto' else tld_defaults.get(voice_gender, 'com')
    slow = speed < 0.8           # gTTS supports slow=True only

    # Generate MP3 with gTTS
    mp3_buf_io = io.BytesIO()
    gTTS(text=text, lang='en', tld=tld, slow=slow).write_to_fp(mp3_buf_io)
    mp3_buf_io.seek(0)
    mp3_buf = mp3_buf_io.read()

    # Try converting MP3 → WAV via soundfile for guaranteed browser playback
    # Falls back to raw MP3 with correct MIME type 'audio/mpeg' if unavailable
    try:
        import soundfile as sf, tempfile as tf2, os as os2
        tmp_mp3 = tf2.NamedTemporaryFile(suffix='.mp3', delete=False)
        tmp_mp3.write(mp3_buf)
        tmp_mp3.close()
        audio_data, sr = sf.read(tmp_mp3.name)
        os2.unlink(tmp_mp3.name)
        wav_out = io.BytesIO()
        sf.write(wav_out, audio_data, sr, format='WAV')
        wav_out.seek(0)
        return wav_out.read(), 'audio/wav'
    except Exception:
        # soundfile cannot decode MP3 — return MP3 with correct MIME type
        return mp3_buf, 'audio/mpeg'


def tts_page():
    """Render the text-to-speech settings page."""
    st.markdown('<div class="main-header">🔊 TTS Settings</div>', unsafe_allow_html=True)

    if not st.session_state.db_connected:
        st.error("⚠️ Database not connected. Please check your Neo4j connection.")
        return

    st.markdown("Configure personalized text-to-speech voice preferences for each speaker.")

    # ── Initialize speaker manager ─────────────────────────────────────────
    if 'speaker_manager' not in st.session_state:
        try:
            from src.speaker import SpeakerProfileManager
            client = get_database_client()
            st.session_state.speaker_manager = SpeakerProfileManager(client)
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            return

    speakers = st.session_state.speaker_manager.get_all_speakers()
    if not speakers:
        st.info("No speakers enrolled yet. Go to the Speaker Identification page to enroll.")
        return

    # ── Speaker selector ───────────────────────────────────────────────────
    st.markdown("### 🎛️ Voice Preferences")
    speaker_names = {s['speaker_id']: s['name'] for s in speakers}
    selected_speaker_id = st.selectbox(
        "Select Speaker",
        options=list(speaker_names.keys()),
        format_func=lambda x: speaker_names[x],
    )

    if not selected_speaker_id:
        return

    # Reload prefs when user switches speaker
    current_prefs = st.session_state.speaker_manager.get_voice_preferences(selected_speaker_id)
    speaker_label = speaker_names[selected_speaker_id]

    st.markdown(f"**Configuring voice for:** {speaker_label}")

    # ── Voice settings UI ──────────────────────────────────────────────────
    GENDER_OPTIONS  = ['female', 'male', 'neutral', 'default']
    ACCENT_OPTIONS  = {
        'US English (female default)':   'com',
        'British English (neutral)':     'co.uk',
        'Australian English':            'com.au',
        'Indian English':                'co.in',
        'Canadian English':              'ca',
        'Auto (match gender)':           'auto',
    }

    col1, col2 = st.columns(2)

    with col1:
        saved_gender = current_prefs.get('voice_gender', 'female')
        if saved_gender not in GENDER_OPTIONS:
            saved_gender = 'female'

        voice_gender = st.selectbox(
            "Voice Gender",
            options=GENDER_OPTIONS,
            index=GENDER_OPTIONS.index(saved_gender),
            help="🎙️ Chooses the voice engine: Male uses your system voice; others use Google TTS.",
        )

        accent_labels = list(ACCENT_OPTIONS.keys())
        saved_accent  = current_prefs.get('accent', 'auto')
        saved_accent_label = next(
            (k for k, v in ACCENT_OPTIONS.items() if v == saved_accent),
            'Auto (match gender)'
        )
        accent_label = st.selectbox(
            "Accent / Region",
            options=accent_labels,
            index=accent_labels.index(saved_accent_label),
            help="Regional accent applied to the voice",
        )
        accent = ACCENT_OPTIONS[accent_label]

        speed = st.slider(
            "Speaking Speed",
            min_value=0.5, max_value=2.0,
            value=float(current_prefs.get('speed', 1.0)),
            step=0.1,
            help="1.0 = normal speed  |  < 0.8 activates slow mode for gTTS",
        )

    with col2:
        pitch = st.slider(
            "Pitch Adjustment",
            min_value=-10.0, max_value=10.0,
            value=float(current_prefs.get('pitch', 0.0)),
            step=0.5,
            help="Stored with preference (affects Coqui voices when available)",
        )

        # Show current saved voice as a summary card
        st.markdown("**Currently saved voice:**")
        saved_gender_disp  = current_prefs.get('voice_gender', '—')
        saved_accent_disp  = current_prefs.get('accent', 'auto')
        saved_accent_label_disp = next(
            (k for k, v in ACCENT_OPTIONS.items() if v == saved_accent_disp),
            saved_accent_disp
        )
        saved_speed_disp   = current_prefs.get('speed', 1.0)
        st.info(
            f"👤 **{speaker_label}**  \n"
            f"🎙️ Gender: `{saved_gender_disp}`  \n"
            f"🌍 Accent: `{saved_accent_label_disp}`  \n"
            f"⚡ Speed: `{saved_speed_disp}x`"
        )

    # ── Preview section ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎧 Preview")

    preview_text = st.text_input(
        "Preview Text",
        value=f"Hello! I am {speaker_label} and this is my personalized voice.",
        key=f"preview_text_{selected_speaker_id}",
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶️ Preview Voice", use_container_width=True):
            with st.spinner("Generating preview audio..."):
                try:
                    audio_bytes, audio_fmt = _generate_preview_audio(
                        text=preview_text,
                        voice_gender=voice_gender,
                        accent=accent,
                        speed=speed,
                    )
                    st.audio(audio_bytes, format=audio_fmt)
                except Exception as e:
                    st.error(f"Preview failed: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

    with col2:
        if st.button("💾 Save Preferences", type="primary", use_container_width=True):
            new_prefs = {
                'voice_gender': voice_gender,
                'accent':       accent,
                'speed':        speed,
                'pitch':        pitch,
                # keep voice_name for backward compat with Coqui
                'voice_name':   voice_gender,
            }
            if st.session_state.speaker_manager.update_voice_preferences(selected_speaker_id, new_prefs):
                st.success(f"✅ Voice preferences saved for **{speaker_label}**!")
                st.balloons()
            else:
                st.error("Failed to save preferences")


def statistics_page():
    """Render the statistics page."""
    st.markdown('<div class="main-header">📊 Database Statistics</div>', unsafe_allow_html=True)
    
    if not st.session_state.db_connected:
        st.error("⚠️ Database not connected. Please check your Neo4j connection.")
        return
    
    client = get_database_client()
    
    with st.spinner("Loading statistics..."):
        stats = client.get_statistics()
    
    # Display statistics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats['quotes']:,}</div>
            <div class="stat-label">Total Quotes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats['authors']:,}</div>
            <div class="stat-label">Authors</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats['works']:,}</div>
            <div class="stat-label">Works</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats['said_relationships']:,}</div>
            <div class="stat-label">Relationships</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Top authors
    st.markdown("### 📖 Most Quoted Sources")
    
    query = """
    MATCH (p)-[:HAS_QUOTE|SAID]->(q:Quote)
    WITH p, count(q) AS quote_count
    ORDER BY quote_count DESC
    LIMIT 10
    RETURN p.name AS author, quote_count
    """
    
    top_authors = client.execute_query(query)
    
    if top_authors:
        for i, author in enumerate(top_authors, 1):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{i}. {author['author']}**")
            with col2:
                st.markdown(f"{author['quote_count']} quotes")
    else:
        st.info("No data available yet. Run the ingestion pipeline to populate the database.")


def main():
    """Main application entry point."""
    initialize_session_state()
    
    # Render sidebar and get selected page
    page = sidebar()
    
    # Render selected page
    if page == "🤖 Chatbot & Search":
        chatbot_page()
    elif page == "🎙️ Voice Chat":
        voice_chat_page()
    elif page == "🎤 Speaker Identification":
        speaker_id_page()
    elif page == "🔊 Text-to-Speech":
        tts_page()



if __name__ == "__main__":
    main()
