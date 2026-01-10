"""
Automatic Speech Recognition (ASR) Module

This module provides speech-to-text conversion using Whisper.
For the actual implementation, see asr_whisper.py.
"""
import logging
from typing import Optional, Dict
from .asr_whisper import ASRWhisper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ASRModule:
    """
    Automatic Speech Recognition module using pre-trained models.
    
    This is a wrapper around ASRWhisper for backward compatibility.
    Use ASRWhisper directly for more control.
    """
    
    def __init__(self, model_type: str = 'whisper', model_size: str = 'base'):
        """
        Initialize the ASR module.
        
        Args:
            model_type: Type of ASR model (only 'whisper' is supported)
            model_size: Model size ('tiny', 'base', 'small', 'medium', 'large-v2')
        """
        if model_type != 'whisper':
            logger.warning(f"Only 'whisper' is supported. Ignoring model_type='{model_type}'")
        
        self.model_type = 'whisper'
        self.model_size = model_size
        self.whisper = ASRWhisper(model_size=model_size)
        
        logger.info(f"ASR Module initialized with Whisper ({model_size})")
    
    def load_model(self):
        """Load the pre-trained ASR model."""
        self.whisper._load_model()
    
    def transcribe(self, audio_path: str, language: str = 'en') -> Dict:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            language: Language code (default: 'en', use None for auto-detect)
            
        Returns:
            Dictionary with transcription and metadata:
            {
                'text': str,
                'language': str,
                'segments': list
            }
        """
        # Use None for auto-detection if language is 'en' (common default)
        lang = None if language == 'en' else language
        return self.whisper.transcribe_file(audio_path, language=lang)
    
    def transcribe_stream(self, audio_stream):
        """
        Transcribe audio stream in real-time.
        
        NOTE: Streaming not implemented yet.
        """
        logger.warning("Streaming transcription not implemented yet")
        raise NotImplementedError("Streaming ASR coming soon")


# For backward compatibility
VoiceFlow = None  # Moved to orchestrator.py


def main():
    """Test the ASR module."""
    print("ASR Module - Whisper Implementation")
    print("=" * 60)
    print("\nThis module provides speech recognition using Whisper.")
    print("\nTo use this module:")
    print("1. Install dependencies: pip install faster-whisper")
    print("2. Provide an audio file path")
    print("3. Call asr.transcribe(audio_path)")
    print("\nExample:")
    print("```python")
    print("asr = ASRModule(model_type='whisper', model_size='base')")
    print("result = asr.transcribe('audio.wav')")
    print("print(result['text'])")
    print("```")


if __name__ == '__main__':
    main()
