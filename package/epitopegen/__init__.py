"""
EpitopeGen: Single-cell TCR-based epitope sequence prediction
"""

from .inference import EpitopeGenPredictor
from .annotate import EpitopeAnnotator,EpitopeEnsembler,visualize_match_overlaps_parallel
from .config import (
    TOKENIZER_PATH,
    MODEL_CHECKPOINTS,
    ZENODO_URL,
    DEFAULT_CHECKPOINT,
    DEFAULT_CACHE_DIR,
    GENES_OF_INTEREST,
    GENE_GROUPS
)
from .analyze_deg import DEGAnalyzer
from .analyze_pa import PARatioAnalyzer
from .analyze_pre import PRERatioAnalyzer
from .analyze_ce import TCRUMAPVisualizer
from .analyze_antigen import AntigenAnalyzer,CoronavirusProteinStandardizer

__version__ = "0.1.0"

__all__ = [
    "EpitopeGenPredictor",
    "EpitopeAnnotator",
    "visualize_match_overlaps_parallel",
    "EpitopeEnsembler",
    "TOKENIZER_PATH",
    "MODEL_CHECKPOINTS",
    "ZENODO_URL",
    "DEGAnalyzer",
    "PARatioAnalyzer",
    "PRERatioAnalyzer"
]
