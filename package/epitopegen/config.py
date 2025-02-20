# package/epitopegen/config.py
import os
from pathlib import Path

# Package resource paths
PACKAGE_ROOT = Path(__file__).parent
RESOURCES_PATH = PACKAGE_ROOT / "resources"
TOKENIZER_PATH = RESOURCES_PATH / "tokenizer"

# Model configuration
DEFAULT_CACHE_DIR = os.path.expanduser("./.cache/epitopegen")
ZENODO_URL = "https://zenodo.org/records/14897624/files/checkpoints.zip"
MODEL_CHECKPOINTS = {
    "ckpt1": "checkpoints/epitopegen_weight_1/epoch_28/pytorch_model.bin",
    "ckpt2": "checkpoints/epitopegen_weight_2/epoch_26/pytorch_model.bin",
    "ckpt3": "checkpoints/epitopegen_weight_3/epoch_19/pytorch_model.bin",
    "ckpt4": "checkpoints/epitopegen_weight_4/epoch_21/pytorch_model.bin",
    "ckpt5": "checkpoints/epitopegen_weight_5/epoch_28/pytorch_model.bin",
    "ckpt6": "checkpoints/epitopegen_weight_6/epoch_28/pytorch_model.bin",
    "ckpt7": "checkpoints/epitopegen_weight_7/epoch_24/pytorch_model.bin",
    "ckpt8": "checkpoints/epitopegen_weight_8/epoch_22/pytorch_model.bin",
    "ckpt9": "checkpoints/epitopegen_weight_9/epoch_24/pytorch_model.bin",
    "ckpt10": "checkpoints/epitopegen_weight_10/epoch_20/pytorch_model.bin",
    "ckpt11": "checkpoints/epitopegen_weight_11/epoch_21/pytorch_model.bin",
}
DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS["ckpt1"]
GENES_OF_INTEREST = [
    'PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'CD244',  # exhaustion markers
    'CD69', 'IL2RA', 'MKI67', 'GZMB', 'PRF1',  # activation and proliferation markers
    'ZAP70', 'LAT', 'LCK',  # TCR pathway that can indicate antigen recognition
    'IFNG', 'TNF',  # cytokines
    'TOX', 'EOMES', 'TBX21',  # exhaustion-assoc TFs
    'SLC2A1', 'PRKAA1',  # metabolic markers
    'HIF1A', 'XBP1',  # stress response
    'CCR7', 'IL7R',  # naive T cells (bystanders)
    'KLRG1', 'CX3CR1',  # terminally differentiated
    'ENTPD1'  # CD39: recently found marker
]
GENE_GROUPS = {
    'Cytotoxicity': ['GZMB', 'PRF1'],
    'Naive/Memory markers': ['CCR7', 'IL7R'],
    'Early activation': ['CD69', 'IL2RA'],
    'Proliferation': ['MKI67'],
    'Exhaustion markers': ['PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'CD244', 'ENTPD1'],
    'Terminal differentiation': ['KLRG1', 'CX3CR1'],
    'TCR signaling': ['ZAP70', 'LAT', 'LCK'],
    'Cytokines': ['IFNG', 'TNF'],
    'Transcription factors': ['TOX', 'EOMES', 'TBX21'],
    'Metabolic regulators': ['SLC2A1', 'PRKAA1'],
    'Stress response': ['HIF1A', 'XBP1']
}
