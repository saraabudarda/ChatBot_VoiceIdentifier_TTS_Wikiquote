"""Voice interaction modules for the Wikiquote NLP system."""

from .asr_whisper import ASRWhisper
from .asr_module import ASRModule
from .tts_module import TTSModule
from .tts_coqui import TTSCoqui, TTSManager

__all__ = [
    'ASRWhisper',
    'ASRModule',
    'TTSModule',
    'TTSCoqui',
    'TTSManager'
]
