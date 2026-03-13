"""
Voice Flow Orchestrator

Coordinates the complete voice interaction pipeline:
User Speech → ASR → Speaker ID → Database Search → Quote Extraction → TTS
"""
import logging
from typing import Dict, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceOrchestrator:
    """
    Complete voice interaction flow orchestrator.
    
    Coordinates all voice modules to provide end-to-end
    voice interaction with the Wikiquote system.
    """
    
    def __init__(
        self,
        asr_module,
        speaker_identifier,
        simple_router,
        response_generator,
        tts_manager
    ):
        """
        Initialize the voice orchestrator.
        
        Args:
            asr_module: ASR module instance (ASRWhisper or ASRModule)
            speaker_identifier: SpeakerIdentifier instance
            simple_router: SimpleRouter instance
            response_generator: ResponseGenerator instance
            tts_manager: TTSManager instance
        """
        self.asr = asr_module
        self.speaker_id = speaker_identifier
        self.simple_router = simple_router
        self.response_generator = response_generator
        self.tts = tts_manager
        
        logger.info("Voice Orchestrator initialized")
    
    def _validate_audio_text(self, display_text: str, audio_text: str) -> bool:
        """
        Validate that audio text matches displayed quote exactly.
        
        This is a critical safety check to prevent:
        - Audio speaking different content than displayed
        - Invented or extended quotes in audio
        - Mismatched attribution
        
        Args:
            display_text: Text shown to user
            audio_text: Text to be spoken
            
        Returns:
            True if texts match, False otherwise
        """
        # Normalize whitespace for comparison
        display_normalized = ' '.join(display_text.split())
        audio_normalized = ' '.join(audio_text.split())
        
        # Check exact match
        if display_normalized == audio_normalized:
            logger.info("✓ Audio text validation passed - exact match")
            return True
        
        # Check if audio is a substring of display (acceptable)
        if audio_normalized in display_normalized:
            logger.warning(f"Audio text is subset of display text - acceptable")
            return True
        
        # Mismatch detected
        logger.error(f"Audio text mismatch!")
        logger.error(f"Display: '{display_normalized[:100]}...'")
        logger.error(f"Audio:   '{audio_normalized[:100]}...'")
        return False
    
    def process_voice_query(self, audio_path: str) -> Dict:
        """
        Process a complete voice query through the full pipeline.
        
        Args:
            audio_path: Path to user's audio query
            
        Returns:
            Dictionary with complete results:
            {
                'transcript': str,
                'language': str,
                'speaker_id': str,
                'speaker_name': str,
                'confidence': float,
                'intent': str,
                'response_text': str,
                'response_audio_path': str,
                'error': str (if any error occurred)
            }
        """
        result = {
            'transcript': '',
            'language': 'unknown',
            'speaker_id': None,
            'speaker_name': 'Unknown',
            'confidence': 0.0,
            'intent': 'search',
            'response_text': '',
            'response_audio_path': None,
            'voice_preferences': None,
            'error': None
        }

        try:
            # ── Step 1: Transcribe speech (ASR) ───────────────────────────
            logger.info("Step 1: Transcribing audio...")
            asr_result = self.asr.transcribe_file(audio_path)
            result['transcript'] = asr_result['text']
            result['language'] = asr_result.get('language', 'en')
            logger.info(f"Transcript: '{result['transcript']}'")

            # ── Step 2: Identify speaker ──────────────────────────────────
            logger.info("Step 2: Identifying speaker...")
            speaker_id, confidence, speaker_data = self.speaker_id.identify_from_file(audio_path)
            result['speaker_id'] = speaker_id
            result['confidence'] = confidence

            if speaker_id:
                result['speaker_name'] = speaker_data.get('name', 'Unknown')
                logger.info(f"Identified: {result['speaker_name']} (confidence: {confidence:.3f})")
            else:
                logger.info(f"Unknown speaker (best match: {confidence:.3f})")

            # ── Step 3: Route query & search database ─────────────────────
            logger.info("Step 3: Routing query and searching database...")
            from src.chatbot.simple_router import RouteType
            route_type, clarification, metadata = self.simple_router.route(result['transcript'])

            if route_type == RouteType.CLARIFICATION:
                result['intent'] = 'clarification'
                result['response_text'] = clarification
                return result

            result['intent'] = 'search'

            # Search quotes
            from src.retrieval.autocomplete import QuoteAutocomplete
            from src.database.neo4j_client import Neo4jClient
            import config

            client = Neo4jClient(
                uri=config.NEO4J_URI,
                user=config.NEO4J_USER,
                password=config.NEO4J_PASSWORD,
                database=config.NEO4J_DATABASE
            )
            try:
                autocomplete = QuoteAutocomplete(client)
                search_results = autocomplete.complete_quote(result['transcript'], max_results=5)
            finally:
                try:
                    client.close()
                except Exception:
                    pass

            # ── Step 4: Build response text ───────────────────────────────
            logger.info("Step 4: Building response...")
            if search_results:
                top = search_results[0]
                quote_text = top.get('quote', '')
                author = top.get('author', 'Unknown')
                work = top.get('work', '')

                from src.chatbot.author_mapper import map_author
                actual_author, _ = map_author(author, work)

                parts = [f'"{quote_text}"\n\n— {actual_author}']
                if work and work != 'Unknown':
                    parts.append(f', from *{work}*')
                if speaker_id:
                    parts.append(f'\n\n*Personalized for {result["speaker_name"]}*')
                result['response_text'] = ''.join(parts)
                audio_text = quote_text  # speak only the quote
            else:
                audio_text = "I couldn't find a relevant quote for your question."
                result['response_text'] = audio_text
                logger.warning("No quotes found in database")

            # ── Step 5: Load speaker voice preferences ────────────────────
            logger.info("Step 5: Loading voice preferences...")
            voice_prefs = {'voice_gender': 'female', 'accent': 'auto', 'speed': 1.0}
            if speaker_id and hasattr(self.speaker_id, 'profile_manager'):
                try:
                    saved = self.speaker_id.profile_manager.get_voice_preferences(speaker_id)
                    if saved:
                        voice_prefs = saved
                        logger.info(f"Voice preferences for {result['speaker_name']}: {voice_prefs}")
                except Exception as vp_err:
                    logger.warning(f"Could not load voice preferences: {vp_err}")

            result['voice_preferences'] = voice_prefs

            # ── Step 6: Generate personalized TTS audio ───────────────────
            logger.info("Step 6: Generating personalized TTS audio...")
            try:
                # Use the same _generate_preview_audio helper used in TTS settings
                import sys, os
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
                from src.ui.streamlit_app import _generate_preview_audio

                audio_bytes, audio_fmt = _generate_preview_audio(
                    text=audio_text,
                    voice_gender=voice_prefs.get('voice_gender', 'female'),
                    accent=voice_prefs.get('accent', 'auto'),
                    speed=float(voice_prefs.get('speed', 1.0))
                )

                # Save to temp file so the orchestrator result path stays compatible
                import tempfile
                suffix = '.wav' if 'wav' in audio_fmt else '.mp3'
                tmp = tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False,
                    dir=tempfile.gettempdir(), prefix='wikiquote_tts_'
                )
                tmp.write(audio_bytes)
                tmp.close()
                result['response_audio_path'] = tmp.name
                result['response_audio_fmt'] = audio_fmt
                logger.info(f"TTS audio saved: {tmp.name} ({len(audio_bytes)} bytes)")

            except Exception as tts_err:
                logger.error(f"TTS synthesis failed: {tts_err}", exc_info=True)
                result['error'] = f"TTS failed: {str(tts_err)}"
                result['response_audio_path'] = None

            logger.info("Voice query processed successfully")
            return result

        except Exception as e:
            logger.error(f"Error processing voice query: {e}", exc_info=True)
            result['error'] = str(e)
            result['response_text'] = "I'm sorry, I encountered an error processing your request."
            return result
    
    def get_pipeline_status(self) -> Dict:
        """
        Get status of all pipeline components.
        
        Returns:
            Dictionary with component status
        """
        status = {
            'asr': {
                'loaded': hasattr(self.asr, 'model') and self.asr.model is not None,
                'info': self.asr.get_model_info() if hasattr(self.asr, 'get_model_info') else {}
            },
            'speaker_id': {
                'registered_speakers': len(self.speaker_id.profile_manager.get_all_speakers()),
                'threshold': self.speaker_id.threshold
            },
            'tts': {
                'loaded': hasattr(self.tts.tts, 'model') and self.tts.tts.model is not None,
                'info': self.tts.tts.get_model_info() if hasattr(self.tts.tts, 'get_model_info') else {}
            }
        }
        
        return status


def main():
    """Test the voice orchestrator."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    print("Voice Orchestrator - Test")
    print("=" * 60)
    print("\nThis module coordinates the full voice interaction pipeline.")
    print("\nPipeline:")
    print("1. ASR: Transcribe speech to text")
    print("2. Speaker ID: Identify who is speaking")
    print("3. Intent Recognition: Understand the query")
    print("4. Database Search: Find relevant quotes")
    print("5. Quote Extraction: Extract the most relevant quote")
    print("6. TTS: Synthesize ONLY the quote (not explanation)")
    print("\nTo use:")
    print("```python")
    print("orchestrator = VoiceOrchestrator(asr, speaker_id, intent, response, tts)")
    print("result = orchestrator.process_voice_query('audio.wav')")
    print("print(result['response_text'])")
    print("# Play result['response_audio_path']")
    print("```")


if __name__ == '__main__':
    main()
