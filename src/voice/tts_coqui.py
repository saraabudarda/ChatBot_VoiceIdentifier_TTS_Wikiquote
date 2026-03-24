"""
Text-to-Speech (TTS) Module - Coqui TTS Implementation

Provides personalized text-to-speech synthesis using Coqui TTS.
"""
import logging
import tempfile
from typing import Optional, Dict
from pathlib import Path
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSCoqui:
    """
    TTS module using Coqui TTS for speech synthesis.
    
    Features:
    - Multi-voice support
    - Fast synthesis
    - High-quality audio output
    - Voice personalization
    """
    
    # Voice gender to model speaker mapping
    VOICE_GENDER_MAP = {
        'male': 'male',
        'female': 'female',
        'neutral': 'default',
        'default': 'default'
    }
    
    def __init__(self, model_name: str = 'tts_models/en/ljspeech/tacotron2-DDC'):
        """
        Initialize the TTS module.
        
        Args:
            model_name: Coqui TTS model name
        """
        self.model_name = model_name
        self.model = None
        self.output_dir = Path(tempfile.gettempdir()) / 'wikiquote_tts'
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"TTS initialized with model: {model_name}")
    
    def _load_model(self):
        """Lazy load the TTS model."""
        if self.model is None:
            try:
                from TTS.api import TTS
                logger.info(f"Loading TTS model: {self.model_name}")
                self.model = TTS(model_name=self.model_name, progress_bar=False)
                logger.info("TTS model loaded successfully")
                
                # Log available voices
                if hasattr(self.model, 'speakers') and self.model.speakers:
                    logger.info(f"Available voices: {self.model.speakers}")
                else:
                    logger.info("Single-speaker model loaded")
                    
            except ImportError:
                logger.error("Coqui TTS not installed. Install with: pip install TTS")
                raise
            except Exception as e:
                logger.error(f"Failed to load TTS model: {e}")
                raise
    
    def _validate_voice_preference(self, voice_preferences: Optional[Dict]) -> Optional[str]:
        """
        Validate and map voice preference to actual voice ID.
        
        Priority:
        1. voice_name (if not 'default') - explicit Voice Style selection
        2. voice_gender - general Voice Gender preference
        
        Args:
            voice_preferences: Voice configuration dict
            
        Returns:
            Validated voice ID or None for default
        """
        if not voice_preferences:
            return None
        
        # Priority 1: voice_name (Voice Style) - if explicitly set
        voice_name = voice_preferences.get('voice_name')
        voice_gender = voice_preferences.get('voice_gender')
        
        # Map gender to voice
        target_voice = None
        
        # Check voice_name first - if it's not 'default', use it
        if voice_name and voice_name != 'default':
            target_voice = voice_name
            logger.info(f"Using explicit voice_name: '{target_voice}'")
        # Otherwise fall back to voice_gender
        elif voice_gender and voice_gender in self.VOICE_GENDER_MAP:
            target_voice = self.VOICE_GENDER_MAP[voice_gender]
            logger.info(f"Voice gender '{voice_gender}' mapped to '{target_voice}'")
        
        # Validate voice exists in model (for multi-speaker models)
        if target_voice and hasattr(self.model, 'speakers') and self.model.speakers:
            available_voices = list(self.model.speakers)
            
            # Try exact match
            if target_voice in available_voices:
                logger.info(f"✓ Voice '{target_voice}' validated")
                return target_voice
            
            # Try case-insensitive match
            for voice in available_voices:
                if voice.lower() == target_voice.lower():
                    logger.info(f"✓ Voice '{voice}' matched (case-insensitive)")
                    return voice
            
            # Voice not found, use default
            logger.warning(f"Voice '{target_voice}' not in available voices: {available_voices}")
            logger.warning("Falling back to default voice")
            return None
        
        return target_voice
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        voice_preferences: Optional[Dict] = None
    ) -> str:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Path to save audio file (optional)
            voice_preferences: Voice configuration:
                {
                    'voice_name': str,  # Voice ID (if multi-speaker model)
                    'speed': float,     # Speaking rate (0.5-2.0)
                    'pitch': float      # Pitch shift (not directly supported)
                }
            
        Returns:
            Path to generated audio file
        """
        self._load_model()
        
        # Generate output path if not provided
        if not output_path:
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            output_path = str(self.output_dir / f"tts_{text_hash}.wav")
        
        logger.info(f"Synthesizing: '{text[:50]}...'")
        logger.info(f"Voice preferences: {voice_preferences}")
        
        try:
            # Validate and get voice
            speaker = self._validate_voice_preference(voice_preferences)
            speed = voice_preferences.get('speed', 1.0) if voice_preferences else 1.0
            
            # Check if model supports multi-speaker
            if hasattr(self.model, 'speakers') and self.model.speakers and speaker:
                # Multi-speaker model with specific speaker
                logger.info(f"Using multi-speaker synthesis with speaker: {speaker}")
                self.model.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker=speaker
                )
            else:
                # Single-speaker model or default voice
                logger.info("Using single-speaker synthesis (default voice)")
                self.model.tts_to_file(text=text, file_path=output_path)
            
            # Apply speed adjustment if needed
            if speed != 1.0:
                output_path = self._adjust_speed(output_path, speed)
            
            logger.info(f"✓ Audio saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _adjust_speed(self, audio_path: str, speed: float) -> str:
        """
        Adjust playback speed of audio file.
        
        Args:
            audio_path: Path to audio file
            speed: Speed multiplier (0.5-2.0)
            
        Returns:
            Path to adjusted audio file
        """
        try:
            import librosa
            import soundfile as sf
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Time-stretch
            audio_stretched = librosa.effects.time_stretch(audio, rate=speed)
            
            # Save to new file
            output_path = audio_path.replace('.wav', f'_speed{speed}.wav')
            sf.write(output_path, audio_stretched, sr)
            
            logger.info(f"Speed adjusted to {speed}x")
            return output_path
            
        except ImportError:
            logger.warning("librosa not installed. Skipping speed adjustment.")
            logger.warning("Install with: pip install librosa")
            return audio_path
        except Exception as e:
            logger.error(f"Speed adjustment failed: {e}")
            return audio_path
    
    def get_available_voices(self) -> list:
        """
        Get list of available voices.
        
        Returns:
            List of voice names/IDs
        """
        self._load_model()
        
        if hasattr(self.model, 'speakers') and self.model.speakers:
            return list(self.model.speakers)
        else:
            return ['default']
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'loaded': self.model is not None,
            'output_dir': str(self.output_dir)
        }


class TTSManager:
    """
    TTS Manager with speaker profile integration.
    
    Automatically applies voice preferences based on speaker profiles.
    """
    
    def __init__(self, tts_engine: TTSCoqui, profile_manager=None):
        """
        Initialize TTS manager.
        
        Args:
            tts_engine: TTSCoqui instance
            profile_manager: SpeakerProfileManager instance (optional)
        """
        self.tts = tts_engine
        self.profile_manager = profile_manager
    
    def synthesize_for_speaker(
        self,
        text: str,
        speaker_id: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Synthesize speech with speaker's voice preferences.
        
        Args:
            text: Text to synthesize
            speaker_id: Speaker identifier
            output_path: Path to save audio (optional)
            
        Returns:
            Path to generated audio file
        """
        # Get voice preferences for speaker
        voice_prefs = None
        if self.profile_manager:
            voice_prefs = self.profile_manager.get_voice_preferences(speaker_id)
            logger.info(f"Using voice preferences for {speaker_id}: {voice_prefs}")
        
        # Synthesize with preferences
        return self.tts.synthesize(text, output_path, voice_prefs)


def main():
    """Test the TTS module."""
    print("TTS Coqui Module - Implementation")
    print("=" * 60)
    
    # Initialize TTS
    tts = TTSCoqui()
    
    print(f"\nModel Info:")
    info = tts.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test synthesis
    print("\nTesting synthesis...")
    try:
        audio_path = tts.synthesize("Hello, this is a test of the text to speech system.")
        print(f"Generated audio: {audio_path}")
        
        # Test with speed adjustment
        audio_path_fast = tts.synthesize(
            "This is faster speech.",
            voice_preferences={'speed': 1.5}
        )
        print(f"Generated fast audio: {audio_path_fast}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Make sure Coqui TTS is installed: pip install TTS")


if __name__ == '__main__':
    main()
