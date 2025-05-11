# nemo_forced_aligner/__init__.py
from .aligner import NeMoForcedAligner
from .constants import SUPPORTED_MODELS

__version__ = "0.1.0"
__all__ = ['NeMoForcedAligner', 'SUPPORTED_MODELS']