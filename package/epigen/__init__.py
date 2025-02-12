"""
SCEpiGen: Single-cell TCR-based epitope sequence prediction
"""

from .inference import EpiGenPredictor
from .annotate import EpitopeAnnotator,EpitopeEnsembler

__version__ = "0.1.0"

__all__ = [
    "EpiGenPredictor",
    "EpitopeAnnotator",
    "EpitopeEnsembler"
]
