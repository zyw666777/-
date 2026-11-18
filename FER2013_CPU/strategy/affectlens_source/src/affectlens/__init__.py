"""AffectLens: uncertainty-aware facial expression recognition."""

from .constants import EMOTIONS
from .model import AffectLensNet, ModelOutput

__all__ = ["AffectLensNet", "EMOTIONS", "ModelOutput"]
__version__ = "0.1.0"

