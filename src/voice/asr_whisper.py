"""
Automatic Speech Recognition (ASR) Module - Whisper Implementation

Uses faster-whisper for efficient speech-to-text conversion.
"""
import logging
import numpy as np
from typing import Dict, Optional, Union
from pathlib import Path
import tempfile
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ASRWhisper:
    """
    ASR module using faster-whisper for speech recognition.
    
    Features:
    - Fast transcription with GPU/CPU support
    - Automatic language detection
    - Audio normalization (16kHz mono)
    - Support for file and array input
    """
    
    def __init__(self, model_size: str = 'base', device: str = 'auto', compute_type: str = 'default'):
        """
        Initialize the ASR module.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v2')
            device: Device to use ('cpu', 'cuda', 'auto')
            compute_type: Compute type ('int8', 'float16', 'default')
        """
        self.model_size = model_size
        self.device = device if device != 'auto' else self._detect_device()
        self.compute_type = compute_type if compute_type != 'default' else self._get_default_compute_type()
        self.model = None
        self.sample_rate = 16000
        
        logger.info(f"ASR initialized: model={model_size}, device={self.device}, compute={self.compute_type}")
    
    def _detect_device(self) -> str:
        """Detect available device (CUDA or CPU)."""
        try:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            return 'cpu'
    
    def _get_default_compute_type(self) -> str:
        """Get default compute type based on device."""
        if self.device == 'cuda':
            return 'float16'
        else:
            return 'int8'
    
    def _load_model(self):
        """Lazy load the Whisper model."""
        if self.model is None:
            try:
                from faster_whisper import WhisperModel
                logger.info(f"Loading Whisper model: {self.model_size}")
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type
                )
                logger.info("Whisper model loaded successfully")
            except ImportError:
                logger.error("faster-whisper not installed. Install with: pip install faster-whisper")
                raise
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise
    
    def transcribe_file(self, audio_path: Union[str, Path], language: Optional[str] = None,
                        initial_prompt: Optional[str] = None) -> Dict:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            language: Language code (e.g., 'en', 'es'). If None, auto-detect.
            initial_prompt: Optional domain hint for Whisper (e.g. "Famous literary quote:")
            
        Returns:
            Dictionary with transcription results:
            {
                'text': str,           # Full transcription
                'language': str,       # Detected/specified language
                'segments': list,      # Segment details (optional)
                'confidence': float    # Average confidence (if available)
            }
        """
        self._load_model()
        
        try:
            audio_path = str(audio_path)
            logger.info(f"Transcribing file: {audio_path}")
            
            # Transcribe with faster-whisper
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                initial_prompt=initial_prompt,
                vad_filter=True,  # Voice activity detection
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Collect segments and build full text
            full_text = []
            segment_list = []
            total_confidence = 0.0
            segment_count = 0
            
            for segment in segments:
                full_text.append(segment.text)
                segment_list.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip()
                })
                # Note: faster-whisper doesn't provide per-segment confidence
                segment_count += 1
            
            result = {
                'text': ' '.join(full_text).strip(),
                'language': info.language,
                'segments': segment_list,
                'confidence': None  # faster-whisper doesn't provide this
            }
            
            logger.info(f"Transcription complete: '{result['text'][:100]}...' (language: {result['language']})")
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def transcribe_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None
    ) -> Dict:
        """
        Transcribe audio array to text.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio
            language: Language code (e.g., 'en', 'es'). If None, auto-detect.
            
        Returns:
            Dictionary with transcription results (same format as transcribe_file)
        """
        self._load_model()
        
        try:
            # Normalize audio
            audio_normalized = self._normalize_audio(audio_array, sample_rate)
            
            # Save to temporary file (faster-whisper requires file input)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                sf.write(tmp_path, audio_normalized, self.sample_rate)
            
            # Transcribe
            result = self.transcribe_file(tmp_path, language=language)
            
            # Clean up
            Path(tmp_path).unlink()
            
            return result
            
        except Exception as e:
            logger.error(f"Array transcription failed: {e}")
            raise
    
    def _normalize_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Normalize audio to 16kHz mono.
        
        Args:
            audio: Audio array (can be mono or stereo)
            sample_rate: Original sample rate
            
        Returns:
            Normalized audio array (16kHz mono)
        """
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            try:
                import librosa
                audio = librosa.resample(
                    audio,
                    orig_sr=sample_rate,
                    target_sr=self.sample_rate
                )
            except ImportError:
                logger.warning("librosa not installed. Skipping resampling.")
                logger.warning("Install with: pip install librosa")
        
        # Normalize amplitude to [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        
        return audio
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_size': self.model_size,
            'device': self.device,
            'compute_type': self.compute_type,
            'sample_rate': self.sample_rate,
            'loaded': self.model is not None
        }


def main():
    """Test the ASR module."""
    print("ASR Whisper Module - Implementation")
    print("=" * 60)
    
    # Initialize ASR
    asr = ASRWhisper(model_size='base', device='auto')
    
    print(f"\nModel Info:")
    info = asr.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with dummy audio
    print("\nTesting with dummy audio (3 seconds)...")
    dummy_audio = np.random.randn(16000 * 3).astype(np.float32)
    
    try:
        result = asr.transcribe_array(dummy_audio)
        print(f"\nTranscription: {result['text']}")
        print(f"Language: {result['language']}")
        print(f"Segments: {len(result['segments'])}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Dummy audio won't produce meaningful transcription.")
        print("Test with real audio file for actual results.")


if __name__ == '__main__':
    main()
