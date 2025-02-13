# package/epitopegen/annotate.py

import os
import numpy as np
import pandas as pd
import Levenshtein
from pathlib import Path
from itertools import combinations_with_replacement, combinations
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any, Union, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


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

    def _calculate_annotation_stats(self, df: pd.DataFrame, top_k: int) -> dict:
        """Calculate annotation statistics with cumulative matches."""
        match_cols = [f'match_{i}' for i in range(top_k) if f'match_{i}' in df.columns]
        protein_cols = [f'ref_protein_{i}' for i in range(top_k) if f'ref_protein_{i}' in df.columns]

        # Basic stats
        stats = {
            "total_tcrs": len(df),
            "total_predictions": len(df) * len(match_cols)
        }

        # Calculate cumulative matches at each k
        cumulative_matches = {}
        for k in range(len(match_cols)):
            # Get columns up to current k
            cols_to_k = [f'match_{i}' for i in range(k + 1)]
            matches_at_k = df[cols_to_k].any(axis=1).sum()
            match_rate = (matches_at_k / len(df)) * 100
            cumulative_matches[k] = {
                "tcrs_matched": int(matches_at_k),
                "match_rate": match_rate
            }

        stats["cumulative_matches"] = cumulative_matches

        # Most common matched proteins (across all positions)
        stats["matched_proteins"] = pd.Series([
            protein for col in protein_cols
            for protein in df[col].dropna()
        ]).value_counts().to_dict()

        return stats

    def annotate_all(
        self,
        predictions_dir: str,
        method: str = 'levenshtein',
        threshold: int = 1,
        top_k: int = 50,
        max_length: int = 9,
        output_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        print(f"\n=== Running Multi-Model Annotation ===")
        pred_files = os.listdir(predictions_dir)
        print(f"• Using {len(pred_files)} model checkpoints")

        for pred_file in pred_files:
            df_pred = pd.read_csv(f"{predictions_dir}/{pred_file}")
            model_idx = int(os.path.splitext(os.path.basename(pred_file))[0].split("_")[1][4:])
            output_path = f"{output_dir}/annotations_{model_idx}.csv"
            self.annotate(
                predictions_df=df_pred,
                method=method,
                threshold=threshold,
                top_k=top_k,
                max_length=max_length,
                output_path=output_path,
            )


    def annotate(
        self,
        predictions_df: pd.DataFrame,
        method: str = 'levenshtein',
        threshold: int = 1,
        top_k: int = 50,
        max_length: int = 9,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Annotate predictions with phenotype associations.

        Args:
            predictions_df: DataFrame from epitopegenPredictor with 'tcr' and 'pred_*' columns
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
        print("\n=== Starting epitopegen Annotation ===")
        print(f"• Method: {method}")
        print(f"• Distance threshold: {threshold}")
        print(f"• Analyzing top {top_k} predictions")
        print(f"• Reference database size: {len(self.database)} epitopes")

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

            # Process each prediction column
        for k, pred_col in enumerate(pred_columns):
            print(f"Processing {pred_col} ...")

            results = self._process_predictions(
                predictions_df[pred_col],
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

        # Calculate statistics
        stats = self._calculate_annotation_stats(predictions_df, top_k)

        # Save results if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            predictions_df.to_csv(output_path, index=False)
            stats['output_path'] = str(output_path)

        # Print summary
        print("\n=== Annotation Summary ===")
        print(f"• Processed {stats['total_tcrs']} TCRs")

        print("\n• Cumulative matches at different k:")
        print(f"  - Top-1: {stats['cumulative_matches'][0]['tcrs_matched']} TCRs "
              f"({stats['cumulative_matches'][0]['match_rate']:.1f}%)")
        for k in [4, 9, 19, 49]:  # Show matches at key points
            if k < len(stats['cumulative_matches']):
                print(f"  - Top-{k+1}: {stats['cumulative_matches'][k]['tcrs_matched']} TCRs "
                      f"({stats['cumulative_matches'][k]['match_rate']:.1f}%)")

        print("\n• Top source proteins (across all positions):")
        for protein, count in sorted(stats['matched_proteins'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {protein}: {count} matches")

        if output_path:
            print(f"\n• Results saved to: {output_path}")
        print("===========================")

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

    def _calculate_ensemble_stats(self, base_df: pd.DataFrame, annotation_files: List[str], top_k: int) -> dict:
        """Calculate comprehensive ensemble statistics."""
        stats = {
            "num_files": len(annotation_files),
            "total_tcrs": len(base_df),
            "threshold": self.threshold,
            "input_files": [Path(f).name for f in annotation_files],
            "cumulative_matches": {},
            "most_common_proteins": {}
        }

        # Calculate cumulative statistics for each k
        for k in range(top_k):
            match_col = f'match_{k}'
            protein_col = f'ref_protein_{k}'

            matches = base_df[match_col].sum()
            match_rate = (matches / len(base_df)) * 100

            # Get protein distribution for this k
            proteins = base_df[protein_col].value_counts().head(5).to_dict()

            stats["cumulative_matches"][k] = {
                "tcrs_matched": int(matches),
                "match_rate": match_rate,
                "top_proteins": proteins
            }

        # Overall statistics
        all_matches = base_df[[f'match_{k}' for k in range(top_k)]].any(axis=1).sum()
        stats["total_matched_tcrs"] = int(all_matches)
        stats["overall_match_rate"] = (all_matches / len(base_df)) * 100

        return stats

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
        print("\n=== Starting epitopegen Ensemble ===")
        print(f"• Processing {len(annotation_files)} annotation files")
        print(f"• Voting threshold: {self.threshold}")
        print(f"• Analyzing top {top_k} predictions")

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
        # Calculate statistics
        stats = self._calculate_ensemble_stats(base_df, annotation_files, top_k)

        # Save results if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            base_df.to_csv(output_path, index=False)
            stats['output_path'] = str(output_path)

        # Print summary
        print("\n=== Ensemble Summary ===")
        print(f"• Processed {stats['total_tcrs']} TCRs")
        print(f"• Input files: {', '.join(stats['input_files'])}")

        print("\n• Cumulative matches after ensemble:")
        print(f"  - Top-1: {stats['cumulative_matches'][0]['tcrs_matched']} TCRs "
              f"({stats['cumulative_matches'][0]['match_rate']:.1f}%)")
        for k in [1, 2, 4, 8]:  # Show matches at key points
            if k < top_k:
                print(f"  - Top-{k+1}: {stats['cumulative_matches'][k]['tcrs_matched']} TCRs "
                      f"({stats['cumulative_matches'][k]['match_rate']:.1f}%)")

        # Show protein distributions at key points
        print("\n• Top proteins at different k:")
        for k in [0, min(4, top_k-1), min(9, top_k-1)]:  # Show for k=0, 5, 10
            print(f"\n  At k={k}:")
            for protein, count in stats['cumulative_matches'][k]['top_proteins'].items():
                print(f"    - {protein}: {count} matches")

        print(f"\n• Overall: {stats['total_matched_tcrs']} TCRs matched "
              f"({stats['overall_match_rate']:.1f}%)")

        if output_path:
            print(f"\n• Results saved to: {output_path}")
        print("===========================")

        return base_df


def analyze_match_overlap(file1_path: str, file2_path: str, top_k: int = 1,
                         tcr_col: str = 'cdr3', total_population: int = None) -> dict:
    """
    Analyze overlap between matched TCRs in two files considering top K matches.
    A TCR is considered matched if any of its match_0 to match_{K-1} equals 1.

    Args:
        file1_path: Path to first CSV file
        file2_path: Path to second CSV file
        top_k: Number of top matches to consider
        tcr_col: Name of column containing TCR sequences
        total_population: Total size of TCR population (if None, uses length of input data)

    Returns:
        Dictionary containing overlap statistics
    """
    # Read CSVs and drop NA values in TCR column
    df1 = pd.read_csv(file1_path).dropna(subset=[tcr_col])
    df2 = pd.read_csv(file2_path).dropna(subset=[tcr_col])

    # Get total population size
    total_tcrs = total_population or len(df1)

    # Create masks for TCRs with matches in top K predictions
    match_cols = [f'match_{k}' for k in range(top_k)]
    df1_match_mask = df1[match_cols].eq(1).any(axis=1)
    df2_match_mask = df2[match_cols].eq(1).any(axis=1)

    # Get matched TCRs
    df1_matched = df1[df1_match_mask]
    df2_matched = df2[df2_match_mask]

    # Get unique TCRs and calculate overlap
    matched_tcrs_1 = set(df1_matched[tcr_col])
    matched_tcrs_2 = set(df2_matched[tcr_col])
    overlap = matched_tcrs_1 & matched_tcrs_2

    # Calculate expected random overlap
    size1, size2 = len(matched_tcrs_1), len(matched_tcrs_2)
    expected_matches = size1 * (size2 / total_tcrs)
    expected_overlap_pct = expected_matches / min(size1, size2) * 100 if min(size1, size2) > 0 else 0

    # Calculate actual overlap percentages
    overlap_pct_1 = (len(df1_matched[df1_matched[tcr_col].isin(overlap)]) / len(df1_matched)) * 100 if len(df1_matched) > 0 else 0
    overlap_pct_2 = (len(df2_matched[df2_matched[tcr_col].isin(overlap)]) / len(df2_matched)) * 100 if len(df2_matched) > 0 else 0

    # Print results
    print(f"\n*** Between {Path(file1_path).name} & {Path(file2_path).name} (top {top_k} matches)")
    print(f"- Total TCRs: {total_tcrs}")
    print(f"- Expected random overlap: {expected_overlap_pct:.2f}%")
    print(f"- File 1 matched TCRs: {size1}")
    print(f"- File 2 matched TCRs: {size2}")
    print(f"- Overlap percentage in File 1: {overlap_pct_1:.2f}%")
    print(f"- Overlap percentage in File 2: {overlap_pct_2:.2f}%")

    return {
        'file1': Path(file1_path).name,
        'file2': Path(file2_path).name,
        'total_tcrs': total_tcrs,
        'file1_matches': size1,
        'file2_matches': size2,
        'overlap_size': len(overlap),
        'overlap_pct_file1': overlap_pct_1,
        'overlap_pct_file2': overlap_pct_2,
        'expected_overlap_pct': expected_overlap_pct
    }

def analyze_pair(args, top_k=1):
    """
    Helper function to analyze a single pair of files.

    Args:
        args (tuple): Tuple containing (file1_path, file2_path)
        top_k (int): Number of top matches to consider
    """
    try:
        file1_path, file2_path = args
        result = analyze_match_overlap(file1_path, file2_path, top_k=top_k, tcr_col='tcr')
        overlap_value = (result['overlap_pct_file1'] + result['overlap_pct_file2']) / 2
        return (Path(file1_path).parts[-1],
                Path(file2_path).parts[-1],
                overlap_value)
    except Exception as e:
        print(f"Error processing {file1_path} and {file2_path}: {str(e)}")
        return None

def visualize_match_overlaps_parallel(files_list, outdir, top_k=1, n_processes=None):
    """
    Generate a heatmap of match overlaps between all pairs of files using multiprocessing.

    Args:
        files_list (list): List of file paths to analyze
        outdir (str): Output directory for saving the visualization
        top_k (int): Number of top matches to consider (default: 1)
        n_processes (int): Number of processes to use (default: None, uses CPU count - 1)
    """
    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)  # Leave one CPU free

    # Generate all pairs of files
    file_pairs = list(combinations_with_replacement(files_list, 2))

    # Create partial function with fixed top_k
    analyze_pair_fixed = partial(analyze_pair, top_k=top_k)

    # Process pairs in parallel
    print(f"Starting parallel processing with {n_processes} processes...")
    with Pool(processes=n_processes) as pool:
        results = pool.map(analyze_pair_fixed, file_pairs)

    # Filter out any failed results
    results = [r for r in results if r is not None]

    # Initialize empty matrix
    file_names = [Path(f).parts[-1] for f in files_list]

    # Initialize matrix with float type instead of default
    overlap_matrix = pd.DataFrame(0.0, index=file_names, columns=file_names)

    # Fill matrix with explicit type conversion
    for file1, file2, value in results:
        overlap_matrix.loc[file1, file2] = float(value)
        overlap_matrix.loc[file2, file1] = float(value)

    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(overlap_matrix,
                annot=True,
                cmap='YlOrRd',
                square=True,
                cbar_kws={'label': 'Overlap %'})

    plt.title(f'TCR Match Overlap Percentages (top_{top_k})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save plot
    output_path = os.path.join(outdir, f'match_overlap_heatmap_top{top_k}.pdf')
    plt.savefig(output_path)
    plt.close()

    print(f"Heatmap saved to: {output_path}")

    return overlap_matrix, file_names
