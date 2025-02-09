# package/epigen/annotate.py

import numpy as np
import pandas as pd
import Levenshtein
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Any, Union, Tuple, List, Optional

class EpitopeAnnotator:
    """Annotator for epitope predictions to identify phenotype associations."""

    def __init__(self, database_path: str):
        """Initialize the annotator with a reference database.

        Args:
            database_path: Path to CSV file containing reference epitopes
                         Must have 'peptide' and 'protein' columns
        """
        self.database = pd.read_csv(database_path)
        self._validate_database()

    def _validate_database(self):
        """Validate the reference database format."""
        required_columns = {'peptide', 'protein'}
        if not all(col in self.database.columns for col in required_columns):
            raise ValueError(
                f"Database must contain columns: {required_columns}. "
                f"Found: {self.database.columns.tolist()}"
            )

    def annotate(
        self,
        predictions_df: pd.DataFrame,
        method: str = 'levenshtein',
        threshold: int = 1,
        top_k: int = 50,
        max_length: int = 9,
        output_path: Optional[str] = None,
        batch_size: int = 1000
    ) -> pd.DataFrame:
        """Annotate predictions with phenotype associations.

        Args:
            predictions_df: DataFrame from EpiGenPredictor with 'tcr' and 'pred_*' columns
            method: Matching method ('levenshtein' or 'substring')
            threshold: Maximum Levenshtein distance for matches
            top_k: Number of top predictions to analyze
            max_length: Maximum length to consider for epitopes
            output_path: Optional path to save results CSV

        Returns:
            Annotated DataFrame with additional columns:
            - match_*: Binary indicators for matches
            - ref_epitope_*: Matching reference epitopes
            - ref_protein_*: Source proteins for matches
        """
        if method not in ['levenshtein', 'substring']:
            raise ValueError("Method must be either 'levenshtein' or 'substring'")

        # Trim predictions to max_length if specified
        pred_columns = [f'pred_{i}' for i in range(top_k) if f'pred_{i}' in predictions_df.columns]
        if not pred_columns:
            raise ValueError("No prediction columns found in DataFrame")

        if max_length:
            for col in pred_columns:
                predictions_df[col] = predictions_df[col].apply(
                    lambda x: x[:max_length] if isinstance(x, str) else x
                )

        # Process in batches to reduce memory usage
        for i in range(0, len(predictions_df), batch_size):
            batch = predictions_df.iloc[i:i+batch_size].copy()

            # Process each prediction column for this batch
            for k, pred_col in enumerate(pred_columns):
                print(f"Processing {pred_col} (batch {i//batch_size + 1})...")

                results = self._process_predictions(
                    batch[pred_col],
                    threshold=threshold,
                    method=method
                )

                # Add results to DataFrame
                match_col = f'match_{k}'
                ref_epi_col = f'ref_epitope_{k}'
                ref_prot_col = f'ref_protein_{k}'

                predictions_df[match_col], predictions_df[ref_epi_col], predictions_df[ref_prot_col] = zip(*results)

        # Save results if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            predictions_df.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")

        return predictions_df

    def _find_match(
        self,
        pred_epitope: str,
        threshold: int,
        method: str
    ) -> Tuple[int, Optional[str], Optional[str]]:
        """Find matches for a single predicted epitope."""
        if not isinstance(pred_epitope, str):
            return 0, None, None

        if method == 'levenshtein':
            for ref_epitope, ref_protein in zip(self.database['peptide'], self.database['protein']):
                if Levenshtein.distance(pred_epitope, ref_epitope) <= threshold:
                    return 1, ref_epitope, ref_protein

        elif method == 'substring':
            for ref_epitope, ref_protein in zip(self.database['peptide'], self.database['protein']):
                if pred_epitope in ref_epitope:
                    return 1, ref_epitope, ref_protein

        return 0, None, None

    def _process_predictions(
        self,
        pred_column: pd.Series,
        threshold: int,
        method: str
    ) -> List[Tuple[int, Optional[str], Optional[str]]]:
        """Process predictions in parallel."""
        # Create a Pool with explicit start method
        with Pool(cpu_count(), maxtasksperchild=100) as pool:
            results = pool.starmap(
                self._find_match,
                [(epitope, threshold, method) for epitope in pred_column]
            )
        return results


class EpitopeEnsembler:
    """Ensemble multiple annotation results to reduce variance."""

    def __init__(self, threshold: float = 0.5):
        """Initialize ensembler.

        Args:
            threshold: Threshold for majority voting (default: 0.5)
        """
        self.threshold = threshold

    @staticmethod
    def _get_most_frequent(series: pd.Series) -> Any:
        """Get most frequent non-null value in series."""
        return series.value_counts().index[0] if not series.isna().all() else np.nan

    def ensemble(
        self,
        annotation_files: List[str],
        output_path: Optional[str] = None,
        top_k: Optional[int] = 32,
    ) -> pd.DataFrame:
        """Ensemble multiple annotation results using majority voting.

        Args:
            annotation_files: List of paths to annotation CSV files
            output_path: Path to save ensembled results
            top_k: How many top predictions to consider (default: 32)

        Returns:
            DataFrame with ensembled annotations
        """
        cutoffs = range(top_k)
        print(f"Ensembling {len(annotation_files)} annotation files...")

        # Read all annotation files
        dfs = [pd.read_csv(f) for f in annotation_files]
        base_df = dfs[0].copy()

        # Process each cutoff
        for k in cutoffs:
            print(f"Processing k={k}...")

            # Create cumulative matches for each file
            cumulative_matches = []
            protein_predictions = []
            epitope_predictions = []

            for df in dfs:
                # Get matches up to k
                match_cols = [f'match_{i}' for i in range(k + 1)]
                cumulative_match = df[match_cols].any(axis=1).astype(int)
                cumulative_matches.append(cumulative_match)

                # Get corresponding proteins and epitopes
                protein_predictions.append(df[f'ref_protein_{k}'])
                epitope_predictions.append(df[f'ref_epitope_{k}'])

            # Stack predictions
            stacked_matches = pd.concat(cumulative_matches, axis=1)
            stacked_proteins = pd.concat(protein_predictions, axis=1)
            stacked_epitopes = pd.concat(epitope_predictions, axis=1)

            # Perform majority voting
            majority_vote = (stacked_matches.mean(axis=1) >= self.threshold).astype(int)

            # Update result columns
            base_df[f"match_{k}"] = majority_vote

            # Set proteins and epitopes based on majority vote
            mask = majority_vote == 1
            base_df[f"ref_protein_{k}"] = np.nan
            base_df[f"ref_epitope_{k}"] = np.nan

            base_df.loc[mask, f"ref_protein_{k}"] = (
                stacked_proteins.loc[mask].apply(self._get_most_frequent, axis=1)
            )
            base_df.loc[mask, f"ref_epitope_{k}"] = (
                stacked_epitopes.loc[mask].apply(self._get_most_frequent, axis=1)
            )

        # Add summary statistics
        if output_path:
            base_df.to_csv(output_path, index=False)
            print(f"\nSaved ensembled results to: {output_path}")

        # Print statistics
        print("\nEnsemble Statistics:")
        print(f"Total sequences: {len(base_df)}")
        for k in cutoffs:
            positives = (base_df[f'match_{k}'] == 1).sum()
            print(f"k={k}: Positive predictions: {positives} "
                  f"({positives/len(base_df)*100:.2f}%)")

        return base_df


if __name__ == "__main__":
    from package.epigen.inference import EpiGenPredictor

    # Predict from TCR sequences
    tcr_sequences = ["CASIPEGGRETQYF", "CAVRATGTASKLTF"]

    model_weights = [
        "checkpoints/EpiGen_cancer/EpiGen_ckpt_3/epoch_28/pytorch_model.bin",
        "checkpoints/EpiGen_cancer/EpiGen_ckpt_4/epoch_28/pytorch_model.bin",
        "checkpoints/EpiGen_cancer/EpiGen_ckpt_5/epoch_24/pytorch_model.bin"
    ]

    # Initialize predictor
    predictor = EpiGenPredictor(
        checkpoint_path="checkpoints/EpiGen_cancer/EpiGen_ckpt_3/epoch_28/pytorch_model.bin",  # pytorch_model.bin
        tokenizer_path="research/regaler/EpiGen"
    )
    # First run multiple independent annotations
    annotator = EpitopeAnnotator("research/cancer_wu/data/tumor_associated_epitopes.csv")
    annotation_files = []

    for i in range(3):  # Run 3 independent annotations
        predictions_df = predictor.predict(tcr_sequences, top_k=4)
        annotated_df = annotator.annotate(
            predictions_df,
            output_path=f"results/annotation_{i}.csv",
            top_k=4
        )
        annotation_files.append(f"results/annotation_{i}.csv")

    # Then ensemble the results
    ensembler = EpitopeEnsembler(threshold=0.5)
    final_results = ensembler.ensemble(
        annotation_files,
        output_path="results/ensemble_results.csv",
        top_k=4
    )

    print(final_results)
