import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from statsmodels.stats.multitest import multipletests
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable


class PRERatioAnalyzer:
    """Class for performing Phenotype Relative Expansion (PRE) ratio analysis.

    This class provides functionality to analyze clonal expansion patterns between
    Phenotype-Associated (PA) and Non-Associated (NA) T cells across different
    cell types and patterns.
    """

    def __init__(
        self,
        cell_types: List[str],
        pattern_names: List[str],
        pattern_descriptions: Dict[str, str],
        patterns_dict: Dict[str, List[str]],
        output_dir: str = "analysis/PRE_ratio_analysis"
    ):
        """Initialize PRE ratio analyzer.

        Args:
            cell_types: List of cell types to analyze.
            pattern_names: List of pattern names to process.
            pattern_descriptions: Dictionary mapping pattern names to descriptions.
            patterns_dict: Dictionary mapping pattern names to lists of site patterns.
            output_dir: Directory path to save analysis results.
                Defaults to "analysis/PRE_ratio_analysis".
        """
        self.cell_types = cell_types
        self.pattern_names = pattern_names
        self.pattern_descriptions = pattern_descriptions
        self.patterns_dict = patterns_dict
        self.output_dir = Path(output_dir)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input parameters.

        Raises:
            ValueError: If any of the required input parameters is empty or invalid.
        """
        if not self.cell_types:
            raise ValueError("cell_types cannot be empty")

        if not self.pattern_names:
            raise ValueError("pattern_names cannot be empty")

        if not self.pattern_descriptions:
            raise ValueError("pattern_descriptions cannot be empty")

        if not self.patterns_dict:
            raise ValueError("patterns_dict cannot be empty")

    def _calculate_clone_sizes(self, data: pd.DataFrame) -> pd.Series:
        """Calculate clone sizes for a group of cells.

        Args:
            data: DataFrame containing TCR sequence information.

        Returns:
            pd.Series: Clone sizes for each unique TCR sequence.
        """
        return data.groupby('cdr3_nt').size()

    def calculate_expansion_ratio(
        self,
        data: pd.DataFrame,
        match_columns: List[str]
    ) -> Tuple[float, float, float, float, int, int]:
        """Calculate expansion ratio between PA and NA cells.

        Args:
            data: DataFrame containing match columns and clone size information.
            match_columns: List of column names to consider for matching.

        Returns:
            tuple: A tuple containing:
                - mean_PA (float): Mean clone size of PA cells
                - mean_NA (float): Mean clone size of NA cells
                - ratio (float): PA/NA ratio
                - p_value (float): Mann-Whitney U test p-value
                - n_PA (int): Number of PA clones
                - n_NA (int): Number of NA clones
        """
        # Verify match columns exist
        if not all(col in data.columns for col in match_columns):
            return np.nan, np.nan, np.nan, np.nan, 0, 0

        # Create PA and NA masks
        PA_mask = data[match_columns].any(axis=1)
        NA_mask = ~data[match_columns].any(axis=1)

        # Calculate clone sizes
        PA_clones = self._calculate_clone_sizes(data[PA_mask])
        NA_clones = self._calculate_clone_sizes(data[NA_mask])

        # Handle empty groups
        if len(PA_clones) == 0 or len(NA_clones) == 0:
            return np.nan, np.nan, np.nan, np.nan, len(PA_clones), len(NA_clones)

        # Calculate statistics
        mean_PA = np.mean(PA_clones)
        mean_NA = np.mean(NA_clones)
        ratio = mean_PA / mean_NA

        # Perform Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(
            PA_clones,
            NA_clones,
            alternative='greater'
        )

        return mean_PA, mean_NA, ratio, p_value, len(PA_clones), len(NA_clones)

    def analyze(
        self,
        mdata: Dict,
        top_k: int = 1,
        per_patient: bool = False,
        debug: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Perform PRE ratio analysis on T cell data.

        Args:
            mdata: Dictionary containing gene expression data with required 'gex' key.
            top_k: Number of top matches to consider. Defaults to 1.
            per_patient: Whether to perform analysis per patient. Defaults to False.
            debug: Whether to include debug information in plots. Defaults to False.

        Returns:
            dict: Dictionary containing analysis results with keys:
                - 'mean_PA': DataFrame of mean PA clone sizes
                - 'mean_NA': DataFrame of mean NA clone sizes
                - 'ratios': DataFrame of PA/NA ratios
                - 'p_values': DataFrame of corrected p-values
                - 'n_PA': DataFrame of PA clone counts
                - 'n_NA': DataFrame of NA clone counts
        """
        df = self._preprocess_data(mdata['gex'].obs)
        results = {}

        if per_patient:
            patients = df['patient'].unique()
            for patient in patients:
                for k in range(1, 1 + top_k):
                    patient_results = self._analyze_subset(
                        df[df['patient'] == patient],
                        k,
                        debug,
                        patient
                    )
                    results[patient] = patient_results

        # Analyze all patients together
        for k in range(1, 1 + top_k):
            all_results = self._analyze_subset(df, k, debug)
        results['all'] = all_results

        return results

    def _analyze_subset(
        self,
        df: pd.DataFrame,
        top_k: int,
        debug: bool,
        patient: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Analyze a subset of the data (either single patient or all patients).

        Args:
            df: DataFrame containing the data subset to analyze.
            top_k: Number of top matches to consider.
            debug: Whether to include debug information in plots.
            patient: Optional patient identifier for per-patient analysis.

        Returns:
            dict: Dictionary containing analysis results for this subset.
        """
        match_columns = [f'match_{i}' for i in range(top_k)]

        # Initialize result containers
        results = {
            'mean_PA': [],
            'mean_NA': [],
            'ratios': [],
            'p_values': [],
            'n_PA': [],
            'n_NA': []
        }

        # Calculate statistics for each pattern and cell type
        for pattern in self.pattern_names:
            row_results = {key: [] for key in results.keys()}

            for cell_type in self.cell_types:
                df_filtered = df[
                    (df['pattern'].isin(self.patterns_dict[pattern])) &
                    (df['ident'] == cell_type)
                ]

                stats = self.calculate_expansion_ratio(df_filtered, match_columns)

                for key, value in zip(results.keys(), stats):
                    row_results[key].append(value)

            for key in results.keys():
                results[key].append(row_results[key])

        # Convert results to DataFrames
        result_dfs = {}
        for key in results.keys():
            df = pd.DataFrame(
                results[key],
                columns=self.cell_types,
                index=self.pattern_names
            )
            result_dfs[key] = df

        # Apply multiple testing correction
        flat_p_values = result_dfs['p_values'].values.flatten()
        rejected, corrected_p_values, _, _ = multipletests(
            flat_p_values,
            method='fdr_bh'
        )
        result_dfs['p_values'] = pd.DataFrame(
            corrected_p_values.reshape(len(self.pattern_names), len(self.cell_types)),
            columns=self.cell_types,
            index=self.pattern_names
        )

        # Create visualization
        self._create_visualization(
            result_dfs,
            debug=debug,
            patient=patient,
            top_k=top_k
        )

        # Save results
        self._save_results(result_dfs, patient, top_k)

        return result_dfs

    def _create_visualization(
        self,
        results: Dict[str, pd.DataFrame],
        debug: bool,
        top_k: int,
        patient: Optional[str] = None
    ):
        """Create visualization of PRE ratio analysis results.

        Args:
            results: Dictionary containing analysis results DataFrames.
            debug: Whether to include debug information in plot.
            top_k: Number of top matches used in analysis.
            patient: Optional patient identifier for per-patient analysis.
        """
        fig, ax = self._setup_plot(results['mean_PA'])

        self._plot_circles(ax, results, debug)
        self._customize_plot(ax, results['ratios'])

        # Save plot
        suffix = f"_{patient}" if patient else "_all_patients"
        output_path = self.output_dir / f'PRE_ratio{suffix}_k{top_k}.pdf'
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def _setup_plot(self, mean_PA_df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
        """Set up the figure and axes for visualization.

        Args:
            mean_PA_df: DataFrame containing mean PA clone sizes.

        Returns:
            tuple: Figure and Axes objects for plotting.
        """
        n_rows, n_cols = mean_PA_df.shape
        cell_size = 0.9
        fig_width = n_cols * cell_size + 2
        fig_height = n_rows * cell_size * 0.9

        return plt.subplots(figsize=(fig_width, fig_height))

    def _plot_circles(
        self,
        ax: plt.Axes,
        results: Dict[str, pd.DataFrame],
        debug: bool
    ):
        """Plot circles representing clone sizes and ratios.

        Args:
            ax: Matplotlib axes object to plot on.
            results: Dictionary containing analysis results.
            debug: Whether to include debug information.
        """
        cmap = plt.cm.RdBu_r
        norm = plt.Normalize(vmin=0.0, vmax=2.0)

        for i, pattern in enumerate(results['mean_PA'].index):
            for j, cell_type in enumerate(results['mean_PA'].columns):
                size = min(0.35, results['mean_PA'].iloc[i, j] / 30)
                ratio = results['ratios'].iloc[i, j]
                color = cmap(norm(ratio))

                circle = plt.Circle(
                    (j + 0.5, i + 0.5),
                    size,
                    facecolor=color,
                    alpha=0.7,
                    linewidth=1,
                    edgecolor='black'
                )
                ax.add_artist(circle)

                self._add_labels(
                    ax, i, j,
                    results['mean_PA'].iloc[i, j],
                    results['p_values'].iloc[i, j],
                    results['n_PA'].iloc[i, j],
                    results['n_NA'].iloc[i, j],
                    size,
                    debug
                )

    def _add_labels(
        self,
        ax: plt.Axes,
        i: int,
        j: int,
        mean_PA: float,
        p_value: float,
        n_PA: int,
        n_NA: int,
        size: float,
        debug: bool
    ):
        """Add text labels to the plot.

        Args:
            ax: Matplotlib axes object to add labels to.
            i, j: Indices for current cell in the plot.
            mean_PA: Mean PA clone size.
            p_value: P-value for this comparison.
            n_PA, n_NA: Number of PA and NA clones.
            size: Size of the circle.
            debug: Whether to include debug information.
        """
        if p_value < 0.05 or debug:
            ax.text(j + 0.35, i + 0.75, f"p={p_value:.3f}", fontsize=5)

        if size >= 6 / 30:
            text_y = i + 0.5
            text_x = j + 0.5
        else:
            text_y = i + 0.3
            text_x = j + 0.7

        if debug:
            text = f'{mean_PA:.2f}, {n_PA}:{n_NA}'
        else:
            text = f'{mean_PA:.2f}'

        ax.text(
            text_x, text_y,
            text,
            ha='center',
            va='center',
            fontsize=7
        )

    def _customize_plot(
        self,
        ax: plt.Axes,
        ratio_df: pd.DataFrame
    ):
        """Customize the appearance of the plot.

        Args:
            ax: Matplotlib axes object to customize.
            ratio_df: DataFrame containing PA/NA ratios.
        """
        ax.set_xlim(0, len(ratio_df.columns))
        ax.set_ylim(0, len(ratio_df.index))

        ax.set_xticks(np.arange(len(ratio_df.columns)) + 0.5)
        ax.set_yticks(np.arange(len(ratio_df.index)) + 0.5)

        ax.set_xticklabels(
            ratio_df.columns,
            rotation=45,
            ha='right',
            fontdict={'fontsize': 7}
        )
        ax.set_yticklabels(
            ratio_df.index,
            fontdict={'fontsize': 7}
        )

        ax.invert_yaxis()

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(vmin=0.0, vmax=2.0))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label('avg clone size (PA) / avg clone size (NA)')
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(7)

        plt.tight_layout()

    def _save_results(
        self,
        results: Dict[str, pd.DataFrame],
        patient: Optional[str] = None,
        top_k: int = 1
    ):
        """Save analysis results to CSV files.

        Args:
            results: Dictionary containing analysis results DataFrames.
            patient: Optional patient identifier for per-patient analysis.
            top_k: Number of top matches used in analysis.

        Notes:
            Saves the following files in the output directory:
            - mean_PA_clone_sizes_{suffix}_k{top_k}.csv
            - clone_size_comparison_p_values_{suffix}_k{top_k}.csv
            where suffix is either the patient ID or 'all_patients'.
        """
        suffix = f"patient_{patient}" if patient else "all_patients"

        # Save mean PA clone sizes
        results['mean_PA'].to_csv(
            self.output_dir / f'mean_PA_clone_sizes_{suffix}_k{top_k}.csv'
        )

        # Save p-values
        results['p_values'].to_csv(
            self.output_dir / f'clone_size_comparison_p_values_{suffix}_k{top_k}.csv'
        )

        # Optionally save additional statistics
        results['ratios'].to_csv(
            self.output_dir / f'clone_size_ratios_{suffix}_k{top_k}.csv'
        )

        results['n_PA'].to_csv(
            self.output_dir / f'PA_clone_counts_{suffix}_k{top_k}.csv'
        )

        results['n_NA'].to_csv(
            self.output_dir / f'NA_clone_counts_{suffix}_k{top_k}.csv'
        )

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input data for PRE ratio analysis.

        Args:
            df: Input DataFrame containing observation data.

        Returns:
            pd.DataFrame: Preprocessed DataFrame containing only CD8+ T cells with TCRs.

        Notes:
            This method filters the data to include only CD8+ T cells that have
            TCR information. Override this method to implement different
            preprocessing steps if needed.
        """
        # Filter for CD8+ T cells with TCRs
        df = df[
            df['cdr3_nt'].notna() &  # Has TCR sequence
            df['ident'].isin(self.cell_types)  # Is one of the specified cell types
        ].copy()

        return df
