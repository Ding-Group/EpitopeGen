# EpiGen

EpiGen is a deep learning model that predicts cognate epitope sequences from T-cell receptor (TCR) sequences. It helps functionally annotate TCRs in single-cell TCR sequencing data by generating potential binding epitopes and identifying phenotype associations.

## Installation

```bash
pip install epigen
```

## Quick Start

```python
from epigen import EpiGenPredictor

# Initialize predictor
predictor = EpiGenPredictor()

# Predict epitopes for TCR sequences
tcrs = ["CASIPEGGRETQYF", "CAVRATGTASKLTF"]
results = predictor.predict(tcrs)
```

## Basic Usage

### 1. Prepare Input Data

Create a CSV file with TCR sequences:
```csv
text,label
CASIPEGGRETQYF,ZZZZZ
CAVRATGTASKLTF,ZZZZZ
```

### 2. Run Predictions

```python
import pandas as pd
from epigen import EpiGenPredictor

# Initialize predictor
predictor = EpiGenPredictor(tokenizer_path="research/regaler/EpiGen")

# Read TCR sequences
tcrs = pd.read_csv("input.csv")["text"].tolist()

# Generate predictions
results = predictor.predict(
    tcr_sequences=tcrs,
    output_path="predictions.csv",
    top_k=50  # num of epitopes to generate
)
```

### 3. Annotate Phenotypes

```python
from epigen import EpitopeAnnotator

# Initialize annotator with reference database
annotator = EpitopeAnnotator("epitopes_db.csv")

# Annotate predictions
results = annotator.annotate(
    predictions_df=results,
    method='substring',
    output_path="annotations.csv"
)
```

## Key Features

- Generate potential epitope sequences for TCRs
- Support for multiple model checkpoints
- Phenotype annotation using reference databases
- Ensemble predictions for robust results
- Built-in analysis tools

## Resources

- [Detailed Tutorial](https://github.com/Regaler/EpiGen/tutorials)
- [Model Checkpoints](https://zenodo.org/records/14853949)
- [Reference Databases](https://zenodo.org/records/14861398)
- [Development Resources](https://zenodo.org/records/14286754)

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12
- transformers ≥ 4.39.0
- pandas ≥ 1.5.0

## Citation

If you use EpiGen in your research, please cite:
```bibtex
@article{epigen2024,
  title={Generating cognate epitope sequences of T-cell receptors with a generative transformer},
  year={2024}
}
```

## License

MIT License

## Support

For questions and issues:
- Open an issue on [GitHub](https://github.com/regaler/epigen/issues)
- Contact: minukma@cs.ubc.cs
