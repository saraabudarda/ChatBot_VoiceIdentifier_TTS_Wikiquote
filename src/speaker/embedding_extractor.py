"""
Voice Embedding Extractor

Extracts speaker embeddings from audio using SpeechBrain.
"""
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceEmbeddingExtractor:
    """
    Extracts voice embeddings from audio files or arrays.
    
    Uses SpeechBrain's pre-trained speaker recognition model
    to generate fixed-size embeddings for speaker identification.
    """
    
    def __init__(self, model_name: str = "speechbrain/spkrec-ecapa-voxceleb"):
        """
        Initialize the embedding extractor.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.model = None
        self.sample_rate = 16000
        
    def _load_model(self):
        """Lazy load the model."""
        if self.model is None:
            try:
                from speechbrain.pretrained import EncoderClassifier
                logger.info(f"Loading model: {self.model_name}")
                self.model = EncoderClassifier.from_hparams(
                    source=self.model_name,
                    savedir="models/speaker_recognition"
                )
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
    
    def extract_from_file(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Extract embedding from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Embedding vector as numpy array
        """
        self._load_model()
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(str(audio_path))
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Extract embedding
            embedding = self.model.encode_batch(waveform)
            
            # Convert to numpy
            embedding_np = embedding.squeeze().cpu().numpy()
            
            logger.info(f"Extracted embedding of shape {embedding_np.shape}")
            return embedding_np
            
        except Exception as e:
            logger.error(f"Failed to extract embedding from {audio_path}: {e}")
            raise
    
    def extract_from_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract embedding from audio array.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Embedding vector as numpy array
        """
        self._load_model()
        
        try:
            # Convert to torch tensor
            waveform = torch.from_numpy(audio_array).float()
            
            # Add channel dimension if needed
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            # Extract embedding
            embedding = self.model.encode_batch(waveform)
            
            # Convert to numpy
            embedding_np = embedding.squeeze().cpu().numpy()
            
            return embedding_np
            
        except Exception as e:
            logger.error(f"Failed to extract embedding from array: {e}")
            raise
    
    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Ensure result is in [0, 1]
        similarity = (similarity + 1) / 2
        
        return float(similarity)


def main():
    """Test the embedding extractor."""
    extractor = VoiceEmbeddingExtractor()
    
    print("Voice Embedding Extractor initialized")
    print(f"Model: {extractor.model_name}")
    print(f"Sample rate: {extractor.sample_rate} Hz")
    
    # Test with dummy audio
    print("\nTesting with dummy audio...")
    dummy_audio = np.random.randn(16000 * 3)  # 3 seconds
    embedding = extractor.extract_from_array(dummy_audio)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding sample: {embedding[:5]}")


if __name__ == '__main__':
    main()
