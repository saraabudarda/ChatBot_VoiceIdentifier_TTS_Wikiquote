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
from src.chatbot.intent_recognizer import IntentRecognizer, Intent
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
            ["🤖 Chatbot & Search", "🎙️ Voice Chat", "🎤 Speaker Identification", "🔊 Text-to-Speech", "📊 Statistics"],
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
    intent_recognizer = get_intent_recognizer()
    response_generator = get_response_generator()
    
    # Example queries
    with st.expander("💡 Example Queries", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Quote Completion:**")
            st.code("To be or not to be")
            st.code("I think therefore I am")
            
            st.markdown("**Find by Author:**")
            st.code("Quotes by Shakespeare")
            st.code("Show me Einstein quotes")
        
        with col2:
            st.markdown("**Attribution:**")
            st.code("Who said 'cogito ergo sum'?")
            st.code("Author of 'to be or not to be'")
            
            st.markdown("**Other:**")
            st.code("Random quote")
            st.code("Quotes about courage")
    
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
                # Recognize intent
                intent_result = intent_recognizer.recognize(prompt)
                intent = intent_result['intent']
                entities = intent_result['entities']
                
                # Query database based on intent
                results = []
                
                if intent == Intent.QUOTE_COMPLETION:
                    partial_quote = entities.get('partial_quote', prompt)
                    results = autocomplete.complete_quote(partial_quote, max_results=5)
                
                elif intent == Intent.QUOTE_ATTRIBUTION:
                    quote = entities.get('quote', prompt)
                    results = autocomplete.complete_quote(quote, max_results=1)
                
                elif intent == Intent.FIND_BY_AUTHOR:
                    author = entities.get('author', '')
                    results = autocomplete.find_by_author(author, limit=5)
                
                elif intent == Intent.FIND_BY_WORK:
                    work = entities.get('work', '')
                    results = autocomplete.find_by_work(work, limit=5)
                
                elif intent == Intent.RANDOM_QUOTE:
                    results = autocomplete.get_random_quotes(count=1)
                
                elif intent == Intent.QUOTE_RECOMMENDATION:
                    topic = entities.get('topic', '')
                    results = autocomplete.complete_quote(topic, max_results=5)
                
                else:
                    # General search
                    results = autocomplete.complete_quote(prompt, max_results=5)
                
                # Generate response
                response = response_generator.generate(intent, results, entities)
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
            from src.speaker import SpeakerProfileManager, VoiceEmbeddingExtractor, SpeakerIdentifier
            st.session_state.speaker_manager = SpeakerProfileManager(client)
            st.session_state.embedding_extractor = VoiceEmbeddingExtractor()
            st.session_state.speaker_identifier = SpeakerIdentifier(
                st.session_state.speaker_manager,
                st.session_state.embedding_extractor,
                threshold=0.75
            )
        except Exception as e:
            st.error(f"Failed to initialize speaker components: {e}")
            st.info("Make sure all dependencies are installed: `pip install speechbrain librosa soundfile`")
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
                with st.expander(f"👤 {speaker['name']} ({speaker['speaker_id']})"):
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
    4. 🔊 Respond with personalized TTS
    """)
    
    # Initialize voice components (lazy loading)
    if 'voice_orchestrator' not in st.session_state:
        try:
            from src.voice import ASRWhisper, TTSCoqui, TTSManager
            from src.voice.orchestrator import VoiceOrchestrator
            from src.speaker import SpeakerProfileManager, VoiceEmbeddingExtractor, SpeakerIdentifier
            
            client = get_database_client()
            intent_recognizer = get_intent_recognizer()
            response_generator = get_response_generator()
            
            # Initialize components
            with st.spinner("Loading voice models... (this may take a minute)"):
                asr = ASRWhisper(model_size='tiny')  # Use tiny for faster loading
                speaker_manager = SpeakerProfileManager(client)
                embedding_extractor = VoiceEmbeddingExtractor()
                speaker_identifier = SpeakerIdentifier(speaker_manager, embedding_extractor, threshold=0.75)
                tts_engine = TTSCoqui()
                tts_manager = TTSManager(tts_engine, speaker_manager)
                
                # Create orchestrator
                st.session_state.voice_orchestrator = VoiceOrchestrator(
                    asr_module=asr,
                    speaker_identifier=speaker_identifier,
                    intent_recognizer=intent_recognizer,
                    response_generator=response_generator,
                    tts_manager=tts_manager
                )
                
                st.session_state.voice_components_loaded = True
                st.success("✅ Voice models loaded!")
        except Exception as e:
            st.error(f"Failed to initialize voice components: {e}")
            st.info("Make sure all dependencies are installed: `pip install faster-whisper TTS speechbrain`")
            return
    
    st.markdown("---")
    
    # Audio input section
    st.markdown("### 🎤 Audio Input")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_audio = st.file_uploader(
            "Upload audio file",
            type=['wav', 'mp3', 'ogg', 'm4a'],
            help="Upload a voice query (e.g., 'Who said to be or not to be?')"
        )
    
    with col2:
        st.markdown("**Or record:**")
        try:
            from audiorecorder import audiorecorder
            audio_bytes = audiorecorder(
                start_prompt="🎤 Start Recording",
                stop_prompt="⏹️ Stop Recording",
                pause_prompt="⏸️ Pause",
                key="voice_chat_recorder"
            )
        except ImportError:
            st.info("Install: `pip install streamlit-audiorecorder`")
            audio_bytes = None
    
    # Process button
    if uploaded_audio or audio_bytes:
        if st.button("🚀 Process Voice Query", type="primary", use_container_width=True):
            # Save audio to temp file
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                if uploaded_audio:
                    tmp.write(uploaded_audio.read())
                elif audio_bytes:
                    # Convert AudioSegment to WAV bytes
                    import io
                    wav_io = io.BytesIO()
                    audio_bytes.export(wav_io, format="wav")
                    tmp.write(wav_io.getvalue())
                tmp_path = tmp.name
            
            try:
                # Process through orchestrator
                with st.spinner("Processing your voice query..."):
                    result = st.session_state.voice_orchestrator.process_voice_query(tmp_path)
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Processing Results")
                
                # Transcript
                st.markdown("#### 1️⃣ Transcription")
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info(f"**You said:** \"{result['transcript']}\"")
                with col2:
                    st.metric("Language", result['language'].upper())
                
                # Speaker identification
                st.markdown("#### 2️⃣ Speaker Recognition")
                if result['speaker_id']:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.success(f"**Identified:** {result['speaker_name']}")
                    with col2:
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                else:
                    st.warning(f"Unknown speaker (best match: {result['confidence']:.1%})")
                    st.caption("💡 Enroll in the Speaker Identification page to get personalized responses!")
                
                # Intent
                st.markdown("#### 3️⃣ Query Understanding")
                st.info(f"**Intent:** {result['intent']}")
                
                # Response
                st.markdown("#### 4️⃣ Response")
                st.markdown(f"**Answer:**\n\n{result['response_text']}")
                
                # Audio playback
                st.markdown("#### 5️⃣ Audio Response")
                if result['response_audio_path'] and os.path.exists(result['response_audio_path']):
                    with open(result['response_audio_path'], 'rb') as audio_file:
                        audio_data = audio_file.read()
                        st.audio(audio_data, format='audio/wav')
                    
                    # Download button
                    st.download_button(
                        label="⬇️ Download Response Audio",
                        data=audio_data,
                        file_name="response.wav",
                        mime="audio/wav"
                    )
                else:
                    st.error("Audio generation failed")
                
                # Debug panel
                with st.expander("🔍 Debug Information"):
                    st.json({
                        'transcript': result['transcript'],
                        'language': result['language'],
                        'speaker_id': result['speaker_id'],
                        'speaker_name': result['speaker_name'],
                        'confidence': f"{result['confidence']:.3f}",
                        'intent': result['intent'],
                        'audio_path': result['response_audio_path'],
                        'error': result.get('error')
                    })
                
            except Exception as e:
                st.error(f"Processing failed: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
            
            finally:
                # Cleanup temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    
    # Example queries
    st.markdown("---")
    with st.expander("💡 Example Voice Queries"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Quote Questions:**")
            st.code("Who said 'to be or not to be'?")
            st.code("Complete this quote: I think therefore")
            st.code("Tell me a quote about courage")
        with col2:
            st.markdown("**Author Queries:**")
            st.code("Show me quotes by Einstein")
            st.code("What did Shakespeare say?")
            st.code("Give me a random quote")


def tts_page():
    """Render the text-to-speech settings page."""
    st.markdown('<div class="main-header">🔊 TTS Settings</div>', unsafe_allow_html=True)
    
    if not st.session_state.db_connected:
        st.error("⚠️ Database not connected. Please check your Neo4j connection.")
        return
    
    st.markdown("""
    Configure personalized text-to-speech voice preferences for each speaker.
    """)
    
    # Initialize components
    if 'speaker_manager' not in st.session_state:
        try:
            from src.speaker import SpeakerProfileManager
            client = get_database_client()
            st.session_state.speaker_manager = SpeakerProfileManager(client)
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            return
    
    # Get all speakers
    speakers = st.session_state.speaker_manager.get_all_speakers()
    
    if not speakers:
        st.info("No speakers enrolled yet. Go to the Speaker Identification page to enroll.")
        return
    
    st.markdown("### 🎛️ Voice Preferences")
    
    # Speaker selection
    speaker_names = {s['speaker_id']: s['name'] for s in speakers}
    selected_speaker_id = st.selectbox(
        "Select Speaker",
        options=list(speaker_names.keys()),
        format_func=lambda x: speaker_names[x]
    )
    
    if selected_speaker_id:
        # Get current preferences
        current_prefs = st.session_state.speaker_manager.get_voice_preferences(selected_speaker_id)
        
        st.markdown(f"**Configuring voice for:** {speaker_names[selected_speaker_id]}")
        
        # Voice settings
        col1, col2 = st.columns(2)
        
        with col1:
            voice_name = st.selectbox(
                "Voice Style",
                options=['default', 'female', 'male', 'neutral'],
                index=['default', 'female', 'male', 'neutral'].index(current_prefs.get('voice_name', 'default'))
            )
            
            speed = st.slider(
                "Speaking Speed",
                min_value=0.5,
                max_value=2.0,
                value=current_prefs.get('speed', 1.0),
                step=0.1,
                help="1.0 = normal speed"
            )
        
        with col2:
            pitch = st.slider(
                "Pitch Adjustment",
                min_value=-10.0,
                max_value=10.0,
                value=current_prefs.get('pitch', 0.0),
                step=0.5,
                help="0 = no adjustment (Note: pitch adjustment may not be supported)"
            )
        
        # Preview
        st.markdown("---")
        st.markdown("### 🎧 Preview")
        
        preview_text = st.text_input(
            "Preview Text",
            value="Hello! This is a preview of my personalized voice."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Preview Voice", use_container_width=True):
                try:
                    from src.voice import TTSCoqui
                    
                    with st.spinner("Generating preview..."):
                        tts = TTSCoqui()
                        audio_path = tts.synthesize(
                            text=preview_text,
                            voice_preferences={
                                'voice_name': voice_name,
                                'speed': speed,
                                'pitch': pitch
                            }
                        )
                        
                        with open(audio_path, 'rb') as f:
                            st.audio(f.read(), format='audio/wav')
                        
                except Exception as e:
                    st.error(f"Preview failed: {e}")
        
        with col2:
            if st.button("💾 Save Preferences", type="primary", use_container_width=True):
                new_prefs = {
                    'voice_name': voice_name,
                    'speed': speed,
                    'pitch': pitch
                }
                
                if st.session_state.speaker_manager.update_voice_preferences(selected_speaker_id, new_prefs):
                    st.success("✅ Preferences saved!")
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
    elif page == "📊 Statistics":
        statistics_page()


if __name__ == "__main__":
    main()
