"""Speaker identification package."""

from .embedding_extractor import VoiceEmbeddingExtractor
from .profile_manager import SpeakerProfileManager
from .identifier import SpeakerIdentifier

__all__ = [
    'VoiceEmbeddingExtractor',
    'SpeakerProfileManager',
    'SpeakerIdentifier'
]
