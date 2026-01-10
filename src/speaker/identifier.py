"""
Speaker Identifier

Identifies speakers from voice input using embedding comparison.
"""
import logging
import numpy as np
from typing import Optional, Dict, Tuple
from .embedding_extractor import VoiceEmbeddingExtractor
from .profile_manager import SpeakerProfileManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeakerIdentifier:
    """
    Identifies speakers from voice input.
    
    Compares input audio embeddings with stored speaker profiles
    to determine speaker identity.
    """
    
    def __init__(
        self,
        profile_manager: SpeakerProfileManager,
        embedding_extractor: Optional[VoiceEmbeddingExtractor] = None,
        threshold: float = 0.75
    ):
        """
        Initialize the speaker identifier.
        
        Args:
            profile_manager: SpeakerProfileManager instance
            embedding_extractor: VoiceEmbeddingExtractor instance (optional)
            threshold: Minimum similarity score for identification (0-1)
        """
        self.profile_manager = profile_manager
        self.embedding_extractor = embedding_extractor or VoiceEmbeddingExtractor()
        self.threshold = threshold
    
    def identify_from_file(self, audio_path: str) -> Tuple[Optional[str], float, Dict]:
        """
        Identify speaker from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (speaker_id, confidence, speaker_data) or (None, 0.0, {})
        """
        try:
            # Extract embedding from audio
            input_embedding = self.embedding_extractor.extract_from_file(audio_path)
            
            # Identify speaker
            return self._identify_from_embedding(input_embedding)
            
        except Exception as e:
            logger.error(f"Failed to identify speaker from file: {e}")
            return None, 0.0, {}
    
    def identify_from_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000
    ) -> Tuple[Optional[str], float, Dict]:
        """
        Identify speaker from audio array.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Tuple of (speaker_id, confidence, speaker_data) or (None, 0.0, {})
        """
        try:
            # Extract embedding from audio
            input_embedding = self.embedding_extractor.extract_from_array(
                audio_array,
                sample_rate
            )
            
            # Identify speaker
            return self._identify_from_embedding(input_embedding)
            
        except Exception as e:
            logger.error(f"Failed to identify speaker from array: {e}")
            return None, 0.0, {}
    
    def _identify_from_embedding(
        self,
        input_embedding: np.ndarray
    ) -> Tuple[Optional[str], float, Dict]:
        """
        Identify speaker from embedding.
        
        Args:
            input_embedding: Input voice embedding
            
        Returns:
            Tuple of (speaker_id, confidence, speaker_data) or (None, 0.0, {})
        """
        # Get all stored embeddings
        stored_embeddings = self.profile_manager.get_all_embeddings()
        
        if not stored_embeddings:
            logger.warning("No speaker profiles found")
            return None, 0.0, {}
        
        # Compare with all stored embeddings
        best_match_id = None
        best_similarity = 0.0
        
        for speaker_id, stored_embedding in stored_embeddings.items():
            similarity = VoiceEmbeddingExtractor.cosine_similarity(
                input_embedding,
                stored_embedding
            )
            
            logger.debug(f"Speaker {speaker_id}: similarity = {similarity:.3f}")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = speaker_id
        
        # Check if best match meets threshold
        if best_similarity >= self.threshold:
            speaker_data = self.profile_manager.get_speaker(best_match_id)
            logger.info(
                f"Identified speaker: {speaker_data['name']} "
                f"(confidence: {best_similarity:.3f})"
            )
            return best_match_id, best_similarity, speaker_data
        else:
            logger.info(
                f"No confident match found (best: {best_similarity:.3f}, "
                f"threshold: {self.threshold})"
            )
            return None, best_similarity, {}
    
    def get_all_similarities(
        self,
        audio_path: str
    ) -> Dict[str, float]:
        """
        Get similarity scores for all speakers.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary mapping speaker_id to similarity score
        """
        try:
            # Extract embedding
            input_embedding = self.embedding_extractor.extract_from_file(audio_path)
            
            # Get all stored embeddings
            stored_embeddings = self.profile_manager.get_all_embeddings()
            
            # Calculate similarities
            similarities = {}
            for speaker_id, stored_embedding in stored_embeddings.items():
                similarity = VoiceEmbeddingExtractor.cosine_similarity(
                    input_embedding,
                    stored_embedding
                )
                similarities[speaker_id] = similarity
            
            return similarities
            
        except Exception as e:
            logger.error(f"Failed to get similarities: {e}")
            return {}


def main():
    """Test the speaker identifier."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    import config
    from src.database.neo4j_client import Neo4jClient
    
    client = Neo4jClient(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE
    )
    
    profile_manager = SpeakerProfileManager(client)
    identifier = SpeakerIdentifier(profile_manager, threshold=0.75)
    
    print("Testing Speaker Identifier\n")
    print(f"Threshold: {identifier.threshold}")
    print(f"Registered speakers: {len(profile_manager.get_all_speakers())}")
    
    # Test with dummy audio
    print("\nTesting with dummy audio...")
    dummy_audio = np.random.randn(16000 * 3)  # 3 seconds
    speaker_id, confidence, speaker_data = identifier.identify_from_array(dummy_audio)
    
    if speaker_id:
        print(f"Identified: {speaker_data['name']} (confidence: {confidence:.3f})")
    else:
        print(f"Unknown speaker (best match: {confidence:.3f})")
    
    client.close()


if __name__ == '__main__':
    main()
