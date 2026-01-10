"""
Voice Flow Orchestrator

Coordinates the complete voice interaction pipeline:
User Speech → ASR → Speaker ID → Chatbot → TTS
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
        intent_recognizer,
        response_generator,
        tts_manager
    ):
        """
        Initialize the voice orchestrator.
        
        Args:
            asr_module: ASR module instance (ASRWhisper or ASRModule)
            speaker_identifier: SpeakerIdentifier instance
            intent_recognizer: IntentRecognizer instance
            response_generator: ResponseGenerator instance
            tts_manager: TTSManager instance
        """
        self.asr = asr_module
        self.speaker_id = speaker_identifier
        self.intent_recognizer = intent_recognizer
        self.response_generator = response_generator
        self.tts = tts_manager
        
        logger.info("Voice Orchestrator initialized")
    
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
            'intent': 'unknown',
            'response_text': '',
            'response_audio_path': '',
            'error': None
        }
        
        try:
            # Step 1: Transcribe speech (ASR)
            logger.info("Step 1: Transcribing audio...")
            asr_result = self.asr.transcribe_file(audio_path)
            result['transcript'] = asr_result['text']
            result['language'] = asr_result.get('language', 'en')
            logger.info(f"Transcript: '{result['transcript']}'")
            
            # Step 2: Identify speaker
            logger.info("Step 2: Identifying speaker...")
            speaker_id, confidence, speaker_data = self.speaker_id.identify_from_file(audio_path)
            result['speaker_id'] = speaker_id
            result['confidence'] = confidence
            
            if speaker_id:
                result['speaker_name'] = speaker_data.get('name', 'Unknown')
                logger.info(f"Identified: {result['speaker_name']} (confidence: {confidence:.3f})")
            else:
                logger.info(f"Unknown speaker (best match: {confidence:.3f})")
            
            # Step 3: Recognize intent
            logger.info("Step 3: Recognizing intent...")
            intent = self.intent_recognizer.recognize(result['transcript'])
            result['intent'] = intent['type']
            logger.info(f"Intent: {result['intent']}")
            
            # Step 4: Generate response
            logger.info("Step 4: Generating response...")
            response_text = self.response_generator.generate(
                query=result['transcript'],
                intent=intent
            )
            
            # Enhance response with speaker context
            if speaker_id:
                response_text = f"Hello {result['speaker_name']}! {response_text}"
            
            result['response_text'] = response_text
            logger.info(f"Response: '{response_text[:100]}...'")
            
            # Step 5: Generate personalized TTS
            logger.info("Step 5: Synthesizing speech...")
            if speaker_id:
                response_audio = self.tts.synthesize_for_speaker(
                    text=response_text,
                    speaker_id=speaker_id
                )
            else:
                # Use default voice for unknown speakers
                response_audio = self.tts.tts.synthesize(response_text)
            
            result['response_audio_path'] = response_audio
            logger.info(f"Audio generated: {response_audio}")
            
            logger.info("Voice query processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing voice query: {e}")
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
    print("4. Response Generation: Generate natural language response")
    print("5. TTS: Synthesize personalized audio response")
    print("\nTo use:")
    print("```python")
    print("orchestrator = VoiceOrchestrator(asr, speaker_id, intent, response, tts)")
    print("result = orchestrator.process_voice_query('audio.wav')")
    print("print(result['response_text'])")
    print("# Play result['response_audio_path']")
    print("```")


if __name__ == '__main__':
    main()
