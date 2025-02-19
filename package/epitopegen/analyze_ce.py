import os
from itertools import combinations
from multiprocessing import Pool
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP
from Bio.Align import substitution_matrices
import math


def calculate_sequence_distance(seq_pair: Tuple[str, str], blosum62_matrix) -> float:
    """Calculate distance between two sequences using BLOSUM62."""
    seq1, seq2 = seq_pair
    return sum(
        blosum62_matrix.get((a, b), blosum62_matrix.get((b, a), -4))
        for a, b in zip(seq1, seq2)
    )


class TCRUMAPVisualizer:
    """Class for visualizing TCR repertoire data in UMAP space.

    This class provides functionality to analyze and visualize TCR sequences
    using UMAP dimensionality reduction, with options to highlight specific
    patterns and analyze clonal expansion.
    """

    def __init__(
        self,
        patterns: List[List[int]],
        pattern_names: List[str],
        pattern_descriptions: Dict[str, str],
        output_dir: str = "analysis/tcr_umap",
        tcr_column: str = "TRB_1_cdr3",
        condition_column: str = "Who Ordinal Scale"
    ):
        """Initialize TCR UMAP visualizer.

        Args:
            patterns: List of lists containing condition values to group.
            pattern_names: List of names corresponding to each pattern.
            pattern_descriptions: Dictionary mapping pattern names to descriptions.
            output_dir: Directory path to save visualization results.
            tcr_column: Column name containing TCR sequences.
            condition_column: Column name containing condition values.
        """
        self.patterns = patterns
        self.pattern_names = pattern_names
        self.pattern_descriptions = pattern_descriptions
        self.output_dir = Path(output_dir)
        self.tcr_column = tcr_column
        self.condition_column = condition_column

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input parameters."""
        if not self.patterns or not self.pattern_names:
            raise ValueError("patterns and pattern_names cannot be empty")

        if len(self.patterns) != len(self.pattern_names):
            raise ValueError("patterns and pattern_names must have same length")

    def _calculate_clone_sizes(self, data: pd.DataFrame) -> pd.Series:
        """Calculate clone sizes for TCRs."""
        return data.groupby('clonotype').size()

    def _get_unique_clonotypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get representative TCRs for each clonotype."""
        return df.groupby('clonotype').first().reset_index()

    def analyze_clone_sizes(
        self,
        df: pd.DataFrame,
        match_columns: List[str],
        reference_pattern: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Analyze clone sizes across different patterns.

        Args:
            df: DataFrame containing TCR data
            match_columns: List of column names indicating matches
            reference_pattern: Pattern to use as reference for statistical tests

        Returns:
            Tuple of DataFrames containing statistics for:
                - background TCRs
                - primary matched TCRs
                - secondary matched TCRs (if applicable)
        """
        results = []

        for pattern, pattern_name in zip(self.patterns, self.pattern_names):
            pattern_df = df[df[self.condition_column].isin(pattern)].copy()

            # Get matched status
            pattern_df['matched'] = pattern_df[match_columns].any(axis=1)

            # Calculate clone sizes for different groups
            background_sizes = (pattern_df[~pattern_df['matched']]
                             .groupby('clonotype')['clone_size']
                             .first()
                             .values)

            matched_sizes = (pattern_df[pattern_df['matched']]
                          .groupby('clonotype')['clone_size']
                          .first()
                          .values)

            results.append({
                'pattern': pattern_name,
                'background_sizes': background_sizes,
                'matched_sizes': matched_sizes
            })

        # Create and return statistical summaries
        return self._create_statistical_summaries(
            results,
            reference_pattern
        )

    def _create_statistical_summaries(
        self,
        results: List[Dict],
        reference_pattern: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create statistical summaries for clone size analysis."""
        def create_summary(tcr_type: str) -> pd.DataFrame:
            stats_data = []

            # Get reference sizes
            ref_idx = (self.pattern_names.index(reference_pattern)
                      if reference_pattern else 0)
            ref_sizes = results[ref_idx][f'{tcr_type}_sizes']

            for result in results:
                current_sizes = result[f'{tcr_type}_sizes']
                log_sizes = [math.log(x, 10) for x in current_sizes] if len(current_sizes) > 0 else []

                stats = self._calculate_size_statistics(
                    current_sizes,
                    log_sizes,
                    ref_sizes,
                    result['pattern']
                )
                stats_data.append(stats)

            stats_df = pd.DataFrame(stats_data)
            self._apply_multiple_testing_correction(stats_df, reference_pattern)

            return stats_df

        return (create_summary('background'),
                create_summary('matched'))

    def visualize_umap(
        self,
        adata,
        match_columns: List[str],
        n_proc: int = 1,
        sample_size: Optional[int] = None,
        primary_color: str = 'red',
        secondary_color: str = 'orange',
        background_color: str = 'lightgray'
    ):
        """Visualize TCRs in UMAP space.

        Args:
            adata: AnnData object containing TCR data
            match_columns: List of columns indicating matches
            n_proc: Number of processes for distance calculation
            sample_size: Optional size to downsample to
            primary_color: Color for primary matched TCRs
            secondary_color: Color for secondary matched TCRs
            background_color: Color for background TCRs
        """
        df = self._preprocess_data(adata.obs)

        # Process each pattern
        for pattern, pattern_name in zip(self.patterns, self.pattern_names):
            self._visualize_pattern(
                df,
                pattern,
                pattern_name,
                match_columns,
                n_proc,
                sample_size,
                primary_color,
                secondary_color,
                background_color
            )

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess TCR data by calculating clone sizes.

        Args:
            df: DataFrame containing TCR data.

        Returns:
            pd.DataFrame: Processed DataFrame with clone sizes added.

        Raises:
            ValueError: If neither 'clonotype' nor 'cdr3_nt' columns are present.

        Notes:
            Clone sizes are calculated based on either 'clonotype' or 'cdr3_nt'
            columns, in that order of preference. If clone sizes are already
            present, returns the DataFrame unchanged.
        """
        if 'clone_size' in df.columns.tolist():
            return df
        else:
            if 'clonotype' in df.columns.tolist():
                def calculate_clone_sizes(data):
                    return data.groupby('clonotype').size()
                # Calculate clone sizes
                clone_sizes = calculate_clone_sizes(df)
                df['clone_size'] = df['clonotype'].map(clone_sizes)
                print("Clone size calculation is finished! The calculation is based on: clonotype")
            elif 'cdr3_nt' in df.columns.tolist():
                def calculate_clone_sizes(data):
                    return data.groupby('cdr3_nt').size()
                # Calculate clone sizes based on CDR3 nucleotide sequences
                clone_sizes = calculate_clone_sizes(df)
                df['clone_size'] = df['cdr3_nt'].map(clone_sizes)
                print("Clone size calculation is finished! The calculation is based on: cdr3_nt")
            else:
                raise ValueError("The dataframe should have column either `clonotype` or `cdr3_nt` to calculate clone_size")
            return df

    def _visualize_pattern(
        self,
        df: pd.DataFrame,
        pattern: List[int],
        pattern_name: str,
        match_columns: List[str],
        n_proc: int,
        sample_size: Optional[int],
        primary_color: str,
        secondary_color: str,
        background_color: str
    ):
        """Create UMAP visualization for a specific pattern."""
        # Filter and optionally sample data
        pattern_df = df[df[self.condition_column].isin(pattern)].copy()
        if sample_size:
            pattern_df = pattern_df.sample(n=min(sample_size, len(pattern_df)))

        # Calculate UMAP coordinates
        coords = self._calculate_umap_coordinates(
            pattern_df[self.tcr_column].unique(),
            n_proc
        )

        # Create visualization
        self._create_umap_plot(
            pattern_df,
            coords,
            pattern_name,
            match_columns,
            primary_color,
            secondary_color,
            background_color
        )

    def _calculate_umap_coordinates(
        self,
        tcrs: np.ndarray,
        n_proc: int
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate UMAP coordinates for TCR sequences."""
        # Calculate distances
        dists = self._calculate_distances(tcrs, n_proc)

        # Normalize and compute UMAP
        scaler = MinMaxScaler()
        dists_normalized = scaler.fit_transform(dists)
        umap = UMAP(metric='precomputed')
        coords_2d = umap.fit_transform(dists_normalized)

        return {tcr: (x, y) for tcr, (x, y) in zip(tcrs, coords_2d)}

    def _calculate_distances(self, peptides: List[str], n_proc: int = 1) -> np.ndarray:
        """Calculate pairwise BLOSUM62 distances between peptide sequences.
        Args:
            peptides: List of peptide sequences to compare.
            n_proc: Number of processes for parallel computation. Defaults to 1.
        Returns:
            np.ndarray: Matrix of pairwise distances.
        Notes:
            Uses BLOSUM62 substitution matrix for sequence comparison.
            If n_proc > 1, computation is parallelized using multiprocessing.
        """
        blosum62_matrix = substitution_matrices.load('BLOSUM62')
        n_peptides = len(peptides)
        distances = np.zeros((n_peptides, n_peptides))

        if n_proc > 1:
            # Parallel computation
            with Pool(processes=n_proc) as pool:
                # Create a partial function with the blosum62_matrix
                distance_func = partial(calculate_sequence_distance,
                                     blosum62_matrix=blosum62_matrix)
                results = pool.map(distance_func, combinations(peptides, 2))

            # Fill the distance matrix
            idx = np.triu_indices(n_peptides, k=1)
            distances[idx] = results
            distances = distances + distances.T
        else:
            # Sequential computation
            for i, j in combinations(range(n_peptides), 2):
                distance = calculate_sequence_distance(
                    (peptides[i], peptides[j]),
                    blosum62_matrix
                )
                distances[i, j] = distance
                distances[j, i] = distance

        return distances

    def _create_umap_plot(
        self,
        df: pd.DataFrame,
        coords: Dict[str, Tuple[float, float]],
        pattern_name: str,
        match_columns: List[str],
        primary_color: str,
        secondary_color: str,
        background_color: str
    ):
        """Create and save UMAP plot."""
        plt.figure(figsize=(12, 10))

        # Add coordinates to DataFrame
        plot_df = df.copy()
        plot_df['x'] = plot_df[self.tcr_column].map(lambda x: coords[x][0])
        plot_df['y'] = plot_df[self.tcr_column].map(lambda x: coords[x][1])

        # Define matched status
        plot_df['matched'] = plot_df[match_columns].any(axis=1)

        # Plot different groups
        self._plot_tcr_groups(
            plot_df,
            primary_color,
            secondary_color,
            background_color
        )

        # Add statistics and legends
        self._add_plot_annotations(plot_df, pattern_name)

        # Save plot
        plt.savefig(
            self.output_dir / f'tcr_umap_{pattern_name}.pdf',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()

    def _plot_tcr_groups(
        self,
        df: pd.DataFrame,
        primary_color: str,
        secondary_color: str,
        background_color: str
    ):
        """Plot different TCR groups with appropriate styling."""
        # Plot background TCRs
        background = self._get_unique_clonotypes(df[~df['matched']])
        plt.scatter(
            background['x'],
            background['y'],
            s=background['clone_size'] * 10,
            alpha=0.5,
            color=background_color,
            label='Background TCRs'
        )

        # Plot matched TCRs
        matched = self._get_unique_clonotypes(df[df['matched']])
        plt.scatter(
            matched['x'],
            matched['y'],
            s=matched['clone_size'] * 10,
            alpha=0.8,
            color=primary_color,
            label='Matched TCRs'
        )

    def _add_plot_annotations(
        self,
        df: pd.DataFrame,
        pattern_name: str
    ):
        """Add annotations, legends, and statistics to plot."""
        unique_df = self._get_unique_clonotypes(df)
        unique_matched = self._get_unique_clonotypes(df[df['matched']])

        # Add statistics text
        stats_text = (
            f"Total unique clonotypes: {len(unique_df)}\n"
            f"Matched unique clonotypes: {len(unique_matched)} "
            f"({len(unique_matched)/len(unique_df)*100:.1f}%)\n"
            f"Mean clone size: {unique_df['clone_size'].mean():.1f}"
        )

        plt.text(
            0.02, 0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            fontsize=18,
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                alpha=0.8
            )
        )

        # Add title and labels
        plt.title(f"{pattern_name} Group", fontsize=30)
        plt.xlabel("UMAP 1", fontsize=16)
        plt.ylabel("UMAP 2", fontsize=16)

        # Add legend
        plt.legend(loc='upper right')

        # Add clone size legend
        legend_sizes = [2, 10, 20, 30, 50]
        legend_elements = [
            plt.scatter(
                [], [],
                s=size * 10,
                color='gray',
                label=f'{size} TCRs'
            ) for size in legend_sizes
        ]

        plt.gca().add_artist(
            plt.legend(
                handles=legend_elements,
                title="Clone Sizes",
                loc='upper left',
                bbox_to_anchor=(1.15, 1)
            )
        )
