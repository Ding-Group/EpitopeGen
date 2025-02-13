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
    """A class for annotating epitope predictions with phenotype associations.

    This annotator matches predicted epitopes against a reference database to identify
    potential protein associations and phenotypes.

    Attributes:
        database: pandas DataFrame containing the reference epitopes and their
            associated proteins.
    """

    def __init__(self, database_path: str):
        """Initializes the annotator with a reference epitope database.

        Args:
            database_path: Path to CSV file containing reference epitopes.
                The CSV must contain at least 'peptide' and 'protein' columns.

        Raises:
            ValueError: If the database file lacks required columns.
            FileNotFoundError: If the database file cannot be found.
        """
        self.database = pd.read_csv(database_path)
        self._validate_database()

    def _validate_database(self):
        """Validates the format of the reference database.

        Checks if the loaded database contains the required columns
        ('peptide' and 'protein').

        Raises:
            ValueError: If any required columns are missing from the database.
        """
        required_columns = {'peptide', 'protein'}
        if not all(col in self.database.columns for col in required_columns):
            raise ValueError(
                f"Database must contain columns: {required_columns}. "
                f"Found: {self.database.columns.tolist()}"
            )

    def _calculate_annotation_stats(self, df: pd.DataFrame, top_k: int) -> dict:
        """Calculates statistics about epitope matches and protein associations.

        Computes various statistics about the matches between predicted epitopes
        and reference database, including cumulative match rates and protein
        distributions.

        Args:
            df: DataFrame containing prediction results with match and protein
                reference columns (match_0, match_1, etc. and ref_protein_0,
                ref_protein_1, etc.).
            top_k: Number of top predictions to consider for statistics.

        Returns:
            dict: Dictionary containing the following statistics:
                - total_tcrs: Number of TCR sequences analyzed
                - total_predictions: Total number of predictions made
                - cumulative_matches: Dict with statistics at each k:
                    - tcrs_matched: Number of TCRs with at least one match
                    - match_rate: Percentage of TCRs with matches
                - matched_proteins: Dict of protein names and their frequencies

        Note:
            Match columns should be boolean indicators of whether each prediction
            matched the reference database.
        """
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
        """Runs annotation on predictions from multiple model checkpoints.

        Processes all prediction files in the specified directory, annotating each
        with phenotype associations and saving results individually.

        Args:
            predictions_dir: Directory containing prediction CSV files from multiple models.
            method: Method for matching epitopes. Either 'levenshtein' or 'substring'
                (default: 'levenshtein').
            threshold: Maximum Levenshtein distance for considering matches (default: 1).
            top_k: Number of top predictions to analyze per TCR (default: 50).
            max_length: Maximum length to consider for epitopes. Longer sequences will be
                trimmed (default: 9).
            output_dir: Directory to save annotation results. If None, results are only
                returned.

        Returns:
            pd.DataFrame: Combined DataFrame containing annotations for all model predictions.

        Note:
            Expects prediction files to be named in format 'predictions_ckptN.csv' where
            N is the checkpoint number.
        """
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
        """Annotates predicted epitopes with phenotype associations from reference database.

        Matches predicted epitopes against the reference database using either Levenshtein
        distance or substring matching, identifying potential protein associations.

        Args:
            predictions_df: DataFrame containing TCR predictions. Must have 'tcr' column
                and 'pred_0' through 'pred_{top_k-1}' columns.
            method: Matching method to use. Either 'levenshtein' or 'substring'
                (default: 'levenshtein').
            threshold: Maximum Levenshtein distance for considering matches (default: 1).
            top_k: Number of top predictions to analyze per TCR (default: 50).
            max_length: Maximum length to consider for epitopes. Longer sequences will be
                trimmed (default: 9).
            output_path: Path to save annotation results CSV. If None, results are only
                returned.

        Returns:
            pd.DataFrame: Original DataFrame with additional annotation columns:
                - match_{i}: Boolean indicating if prediction i matched reference database
                - ref_epitope_{i}: Matching reference epitope for prediction i
                - ref_protein_{i}: Source protein for matching reference epitope i

        Raises:
            ValueError: If invalid method specified or no prediction columns found.

        Note:
            Prints detailed statistics about matches and protein associations, including
            cumulative match rates at different k values and top matched proteins.
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
        """Finds matching reference epitopes for a single predicted epitope.

        Searches the reference database for matches using either Levenshtein distance
        or substring matching.

        Args:
            pred_epitope: Predicted epitope sequence to find matches for.
            threshold: Maximum Levenshtein distance for considering matches when using
                'levenshtein' method.
            method: Matching method to use, either 'levenshtein' or 'substring'.

        Returns:
            tuple: A tuple containing:
                - int: 1 if match found, 0 otherwise
                - Optional[str]: Matching reference epitope if found, None otherwise
                - Optional[str]: Source protein of matching epitope if found, None otherwise

        Note:
            For 'levenshtein' method, returns first match within threshold distance.
            For 'substring' method, returns first reference epitope containing the
            predicted sequence.
        """
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
        """Processes multiple predictions in parallel using multiprocessing.

        Applies _find_match to each prediction in the input series using a process pool
        for parallel execution.

        Args:
            pred_column: Series of predicted epitope sequences to process.
            threshold: Maximum Levenshtein distance for considering matches when using
                'levenshtein' method.
            method: Matching method to use, either 'levenshtein' or 'substring'.

        Returns:
            list[tuple]: List of tuples, each containing:
                - int: 1 if match found, 0 otherwise
                - Optional[str]: Matching reference epitope if found, None otherwise
                - Optional[str]: Source protein of matching epitope if found, None otherwise

        Note:
            Uses multiprocessing.Pool with maxtasksperchild=100 to prevent memory
            buildup during parallel processing.
        """
        # Create a Pool with explicit start method
        with Pool(cpu_count(), maxtasksperchild=100) as pool:
            results = pool.starmap(
                self._find_match,
                [(epitope, threshold, method) for epitope in pred_column]
            )
        return results


class EpitopeEnsembler:
    """A class for ensembling multiple epitope annotation results to reduce variance.

    This class provides methods for combining and analyzing results from multiple
    model predictions through majority voting and statistical analysis.

    Attributes:
        threshold: Float threshold for majority voting (between 0 and 1).
    """
    def __init__(self, threshold: float = 0.5):
        """Initializes the ensembler with specified voting threshold.

        Args:
            threshold: Minimum fraction of votes needed for consensus in majority
                voting (default: 0.5). Must be between 0 and 1.
        """
        self.threshold = threshold

    @staticmethod
    def _get_most_frequent(series: pd.Series) -> Any:
        """Gets the most frequent non-null value in a pandas Series.

        Args:
            series: Pandas Series containing values to analyze.

        Returns:
            Any: Most frequent non-null value in the series. Returns np.nan if
                all values are null or series is empty.

        Note:
            In case of ties, returns the first value encountered.
        """
        return series.value_counts().index[0] if not series.isna().all() else np.nan

    def _calculate_ensemble_stats(self, base_df: pd.DataFrame, annotation_files: List[str], top_k: int) -> dict:
        """Calculates comprehensive statistics for ensemble predictions.

        Analyzes the ensemble results to generate statistics about matches,
        match rates, and protein distributions at different k values.

        Args:
            base_df: DataFrame containing the ensemble prediction results.
            annotation_files: List of paths to original annotation files used
                in ensemble.
            top_k: Number of top predictions to analyze per TCR.

        Returns:
            dict: Dictionary containing ensemble statistics:
                - num_files: Number of annotation files used
                - total_tcrs: Total number of TCR sequences
                - threshold: Majority voting threshold used
                - input_files: Names of input annotation files
                - cumulative_matches: Dict with statistics at each k:
                    - tcrs_matched: Number of TCRs with matches
                    - match_rate: Percentage of TCRs with matches
                    - top_proteins: Most common matched proteins
                - total_matched_tcrs: Total TCRs with any matches
                - overall_match_rate: Overall percentage of TCRs matched

        Note:
            Statistics are calculated both per-k and across all k values
            to provide comprehensive analysis of ensemble performance.
        """
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
        """Combines multiple annotation results using majority voting ensemble method.

        Processes multiple annotation files to create a consensus prediction through
        majority voting at each k value. For matched sequences, determines the most
        frequent protein and epitope annotations.

        Args:
            annotation_files: List of paths to annotation CSV files. Each file should
                contain matching results from a different model checkpoint.
            output_path: Path to save the ensembled results CSV. If None, results
                are only returned as DataFrame.
            top_k: Number of top predictions to consider per TCR (default: 32).
                Each k represents a different cutoff point for analysis.

        Returns:
            pd.DataFrame: DataFrame containing ensembled results with columns:
                - Original TCR sequence columns
                - match_{k}: Binary indicators for consensus matches at each k
                - ref_protein_{k}: Most frequent matching protein at each k
                - ref_epitope_{k}: Most frequent matching epitope at each k

        Note:
            - Uses the threshold specified during initialization for majority voting
            - Prints detailed statistics about match rates and protein distributions
              at different k values
            - The first annotation file's structure is used as the base for the
              ensemble results
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
    """Analyzes the overlap between matched TCRs in two prediction result files.

    Compares two sets of TCR prediction results to determine the overlap in their
    matches, considering the top K predictions for each TCR. A TCR is considered
    matched if any of its matches from match_0 to match_{K-1} equals 1.

    Args:
        file1_path: Path to first CSV file containing TCR prediction results.
        file2_path: Path to second CSV file containing TCR prediction results.
        top_k: Number of top predictions to consider for each TCR (default: 1).
        tcr_col: Name of the column containing TCR sequences (default: 'cdr3').
        total_population: Total size of TCR population for calculating expected
            overlap. If None, uses the length of input data (default: None).

    Returns:
        dict: Dictionary containing overlap statistics:
            - file1: Name of first input file
            - file2: Name of second input file
            - total_tcrs: Total number of TCRs in population
            - file1_matches: Number of matched TCRs in first file
            - file2_matches: Number of matched TCRs in second file
            - overlap_size: Number of TCRs matched in both files
            - overlap_pct_file1: Percentage of file1 matches also in file2
            - overlap_pct_file2: Percentage of file2 matches also in file1
            - expected_overlap_pct: Expected random overlap percentage

    Note:
        - Input CSV files must contain a TCR sequence column and match_k columns
          (match_0 through match_{K-1})
        - Prints detailed overlap statistics to stdout
        - Handles edge cases where no matches are found in either file
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
    """Analyzes match overlap between a single pair of prediction files.

    Helper function designed to work with multiprocessing, processing a pair of
    files and calculating their overlap statistics.

    Args:
        args: Tuple containing paths to two prediction files (file1_path, file2_path).
        top_k: Number of top predictions to consider for each TCR (default: 1).

    Returns:
        Optional[Tuple[str, str, float]]: If successful, returns tuple containing:
            - str: Filename of first file
            - str: Filename of second file
            - float: Average overlap percentage between the files
        Returns None if processing fails.

    Note:
        - Uses 'tcr' as the column name for TCR sequences
        - Prints error message to stdout if processing fails
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
    """Generates a heatmap visualization of match overlaps between prediction files.

    Creates a heatmap showing the pairwise overlap percentages between all
    combinations of prediction files using parallel processing for efficiency.

    Args:
        files_list: List of paths to prediction result files to analyze.
        outdir: Directory path where the heatmap visualization will be saved.
        top_k: Number of top predictions to consider for each TCR (default: 1).
        n_processes: Number of parallel processes to use. If None, uses CPU count
            minus 1 (default: None).

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Matrix of overlap percentages between all file pairs
            - list[str]: List of filenames used as matrix labels

    Note:
        - Creates a heatmap visualization saved as 'match_overlap_heatmap_topK.pdf'
        - Uses seaborn for heatmap generation with YlOrRd colormap
        - Processes file pairs in parallel for improved performance
        - Creates output directory if it doesn't exist
        - Matrix is symmetric with diagonal values representing self-overlap
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
