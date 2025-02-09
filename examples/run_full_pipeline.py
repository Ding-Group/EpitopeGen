# examples/run_full_pipeline.py

import torch
from pathlib import Path
import pandas as pd
from contextlib import contextmanager
import gc
import os
# Set this at the very beginning of the script
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from package.epigen.inference import EpiGenPredictor
from package.epigen.annotate import EpitopeAnnotator, EpitopeEnsembler

@contextmanager
def predictor_context(checkpoint_path, tokenizer_path, **kwargs):
    """Context manager for EpiGenPredictor to ensure proper resource cleanup."""
    predictor = EpiGenPredictor(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        **kwargs
    )
    try:
        yield predictor
    finally:
        # Clean up GPU memory
        if hasattr(predictor, 'model'):
            del predictor.model
        if hasattr(predictor, 'tokenizer'):
            del predictor.tokenizer
        torch.cuda.empty_cache()
        gc.collect()

def run_predictions(tcr_sequences, model_weights, tokenizer_path, output_dir, top_k=4):
    """Run predictions using multiple model weights."""
    prediction_files = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, weight_path in enumerate(model_weights):
        print(f"\nRunning predictions with model {i+1}/{len(model_weights)}")
        output_path = output_dir / f"predictions_{i}.csv"

        # Use context manager for predictor
        with predictor_context(weight_path, tokenizer_path) as predictor:
            predictions_df = predictor.predict(
                tcr_sequences,
                num_predictions=top_k
            )
            predictions_df.to_csv(output_path, index=False)
            prediction_files.append(output_path)

        print(f"Saved predictions to {output_path}")

    return prediction_files

def run_annotations(prediction_files, database_path, output_dir, top_k=4):
    """Run annotations for all prediction files."""
    annotation_files = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize annotator (can be reused)
    annotator = EpitopeAnnotator(database_path)

    for i, pred_file in enumerate(prediction_files):
        print(f"\nRunning annotations for model {i+1}/{len(prediction_files)}")
        output_path = output_dir / f"annotation_{i}.csv"

        # Load predictions and annotate
        predictions_df = pd.read_csv(pred_file)

        # Explicitly close any CUDA resources before multiprocessing
        torch.cuda.empty_cache()

        annotated_df = annotator.annotate(
            predictions_df,
            output_path=output_path,
            top_k=top_k
        )
        annotation_files.append(output_path)

    return annotation_files

def main():
    # Set number of threads for numpy operations
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    # Configuration
    CONFIG = {
        'model_weights': [
            "checkpoints/EpiGen_cancer/EpiGen_ckpt_3/epoch_28/pytorch_model.bin",
            "checkpoints/EpiGen_cancer/EpiGen_ckpt_4/epoch_28/pytorch_model.bin",
            "checkpoints/EpiGen_cancer/EpiGen_ckpt_5/epoch_24/pytorch_model.bin"
        ],
        'tokenizer_path': "research/regaler/EpiGen",
        'database_path': "research/cancer_wu/data/tumor_associated_epitopes.csv",
        'output_base_dir': "results",
        'top_k': 4
    }

    # Sample TCR sequences
    tcr_sequences = [
        "CASIPEGGRETQYF",
        "CAVRATGTASKLTF"
    ]

    # Create output directories
    base_dir = Path(CONFIG['output_base_dir'])
    pred_dir = base_dir / "predictions"
    annot_dir = base_dir / "annotations"
    base_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Run predictions with multiple models
        print("\n=== Running Predictions ===")
        prediction_files = run_predictions(
            tcr_sequences,
            CONFIG['model_weights'],
            CONFIG['tokenizer_path'],
            pred_dir,
            CONFIG['top_k']
        )

        # Step 2: Run annotations for all predictions
        print("\n=== Running Annotations ===")
        annotation_files = run_annotations(
            prediction_files,
            CONFIG['database_path'],
            annot_dir,
            CONFIG['top_k']
        )

        # Step 3: Ensemble results
        print("\n=== Creating Ensemble ===")
        ensembler = EpitopeEnsembler(threshold=0.5)
        final_results = ensembler.ensemble(
            annotation_files,
            output_path=base_dir / "ensemble_results.csv",
            top_k=CONFIG['top_k']
        )

        # Print final statistics
        print("\n=== Final Results ===")
        print(f"Total sequences analyzed: {len(final_results)}")

        return final_results

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    final_results = main()
