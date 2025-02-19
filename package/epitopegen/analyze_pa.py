import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from statsmodels.stats.multitest import multipletests
from scipy.stats import fisher_exact


class PARatioAnalyzer:
    """Class for performing Phenotype-Association Ratio analysis.

    This class provides functionality to analyze Phenotype-Association (PA) ratios
    across different cell types and patterns.
    """
    def __init__(
        self,
        cell_types: List[str],
        pattern_names: List[str],
        pattern_descriptions: Dict[str, str],
        patterns_dict: Optional[List[List[str]]] = None,
        output_dir: str = "analysis/PA_ratio_analysis"
    ):
        """Initialize PA ratio analyzer.

        Args:
            cell_types: List of cell types to analyze.
            pattern_names: List of pattern names to process.
            pattern_descriptions: Dictionary mapping pattern names to their descriptions.
            patterns_dict: Optional list of lists containing site patterns to analyze.
                Defaults to None.
            output_dir: Directory path to save analysis results.
                Defaults to "analysis/PA_ratio_analysis".
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
            ValueError: If any of the required input parameters (cell_types,
                pattern_names, or pattern_descriptions) is empty.
        """
        if not self.cell_types:
            raise ValueError("cell_types cannot be empty")

        if not self.pattern_names:
            raise ValueError("pattern_names cannot be empty")

        if not self.pattern_descriptions:
            raise ValueError("pattern_descriptions cannot be empty")

    def calculate_pa_ratio_with_stats(
        self,
        data: pd.DataFrame,
        reference_data: pd.DataFrame,
        match_columns: List[str]
    ) -> Tuple[float, float, int]:
        """Calculate PA ratio and perform statistical comparison with reference data.

        Args:
            data: DataFrame containing the current pattern data.
            reference_data: DataFrame containing the reference pattern data
                (naive/healthy).
            match_columns: List of column names to consider for matching.

        Returns:
            tuple: A tuple containing:
                - ratio (float): The calculated PA ratio
                - p_value (float): P-value from Fisher's exact test
                - PA_count (int): Number of PA cells

        Notes:
            If match_columns is None, defaults to ['match_0', 'match_1', ..., 'match_9'].
            Returns (nan, nan, 0) if no matching columns are found in the data.
        """
        if match_columns is None:
            match_columns = [f'match_{i}' for i in range(10)]

        existing_columns = [col for col in match_columns if col in data.columns]
        if not existing_columns:
            return np.nan, np.nan, 0

        # Calculate counts for current pattern
        PA_current = data[existing_columns].any(axis=1).sum()
        total_current = data.shape[0]

        # Calculate counts for reference pattern
        PA_ref = reference_data[existing_columns].any(axis=1).sum()
        total_ref = reference_data.shape[0]

        # Calculate ratio
        ratio = PA_current / total_current if total_current > 0 else np.nan

        # Perform Fisher's exact test
        if total_current > 0 and total_ref > 0:
            contingency_table = [
                [PA_current, total_current - PA_current],
                [PA_ref, total_ref - PA_ref]
            ]
            _, p_value = fisher_exact(contingency_table, alternative='greater')
        else:
            p_value = np.nan

        return ratio, p_value, PA_current

    def analyze(
        self,
        mdata: Dict,
        top_k: int = 8,
        per_patient: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Perform PA ratio analysis on gene expression data.

        Analyzes Phenotype-Association ratios across different patterns and cell types,
        calculating statistics and generating visualizations.

        Args:
            mdata: Dictionary containing gene expression data, with a required 'gex'
                key containing observation data.
            top_k: Number of top matches to consider in the analysis.
                Defaults to 8.
            per_patient: Whether to perform analysis separately for each patient.
                Defaults to False.

        Returns:
            dict: A dictionary with keys as generation numbers (1 to top_k) and values
                as dictionaries containing:
                - 'ratios': DataFrame of PA ratios
                - 'p_values': DataFrame of corrected p-values
                - 'num_cells': DataFrame of cell counts

        Notes:
            The function performs multiple testing correction on p-values and
            generates visualizations for each pattern group.
        """
        df = self._preprocess_data(mdata['gex'].obs)
        results = {}

        for n_gen in range(1, 1 + top_k):
            # Calculate ratios and stats
            ratio_df, p_value_df, num_df = self._calculate_statistics(df, n_gen)

            # Apply multiple testing correction
            corrected_p_value_df = self._apply_multiple_testing_correction(p_value_df)

            # Map pattern names to descriptions
            ratio_df.index = ratio_df.index.map(self.pattern_descriptions)
            corrected_p_value_df.index = corrected_p_value_df.index.map(self.pattern_descriptions)

            # Create visualizations for each pattern group
            self._create_visualizations(
                ratio_df,
                corrected_p_value_df,
                self.pattern_names,
                n_gen
            )

            # Save results
            self._save_results(ratio_df, corrected_p_value_df, num_df, n_gen)

            # Store results for return
            results[n_gen] = {
                'ratios': ratio_df,
                'p_values': corrected_p_value_df,
                'num_cells': num_df
            }

        return results

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input data for PA ratio analysis.

        Args:
            df: Input DataFrame containing observation data from gene expression
                analysis.

        Returns:
            pd.DataFrame: Preprocessed DataFrame ready for PA ratio analysis.

        Notes:
            This is a placeholder method - implement actual preprocessing logic
            based on specific requirements.
        """
        # Implement your preprocessing logic here
        # This is a placeholder - replace with actual implementation
        return df

    def _calculate_statistics(
        self,
        df: pd.DataFrame,
        n_gen: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Calculate PA ratios and statistical measures for all patterns and cell types.

        Args:
            df: Preprocessed DataFrame containing pattern and cell type information.
            n_gen: Number of generations/matches to consider in the analysis.

        Returns:
            tuple: A tuple containing three DataFrames:
                - ratio_df (pd.DataFrame): PA ratios for each pattern-cell type pair
                - p_value_df (pd.DataFrame): Statistical p-values for each pair
                - num_df (pd.DataFrame): Number of cells for each pair

        Notes:
            - The function processes each pattern-cell type combination separately
            - Uses match columns named as 'match_0', 'match_1', etc. up to n_gen-1
            - All resulting DataFrames have patterns as index and cell types as columns
        """
        aggregated_data = []
        p_values = []
        num_data = []

        for pattern in self.pattern_names:
            ratio_row = []
            num_row = []
            p_value_row = []

            for cell_type in self.cell_types:
                # Get data for current pattern and cell type
                df_filtered = df[df['pattern'].isin(self.patterns_dict[pattern]) & (df['ident'] == cell_type)]
                # Get 'All' pattern data for this cell type
                df_all = df[df['ident'] == cell_type]

                match_cols = [f'match_{i}' for i in range(n_gen)]
                ratio, p_value, num_cells = self.calculate_pa_ratio_with_stats(
                    df_filtered, df_all, match_cols
                )

                ratio_row.append(ratio)
                p_value_row.append(p_value)
                num_row.append(num_cells)

            aggregated_data.append(ratio_row)
            p_values.append(p_value_row)
            num_data.append(num_row)

        # Create DataFrames
        ratio_df = pd.DataFrame(aggregated_data,
                              columns=self.cell_types,
                              index=self.pattern_names)
        p_value_df = pd.DataFrame(p_values,
                                columns=self.cell_types,
                                index=self.pattern_names)
        num_df = pd.DataFrame(num_data,
                            columns=self.cell_types,
                            index=self.pattern_names)

        return ratio_df, p_value_df, num_df

    def _apply_multiple_testing_correction(
        self,
        p_value_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply FDR-BH multiple testing correction to p-values.

        Args:
            p_value_df: DataFrame containing p-values with patterns as index and
                cell types as columns.

        Returns:
            pd.DataFrame: DataFrame with corrected p-values, maintaining the same
                structure as input (patterns as index, cell types as columns).

        Notes:
            - Uses Benjamini-Hochberg FDR correction method
            - Applies correction separately for each cell type
            - Maintains original pattern and cell type ordering
        """
        corrected_p_values = []
        for col in range(p_value_df.shape[1]):
            _, corrected, _, _ = multipletests(
                p_value_df.iloc[:, col].values,
                method='fdr_bh'
            )
            corrected_p_values.append(corrected)

        return pd.DataFrame(
            np.array(corrected_p_values).T,
            columns=self.cell_types,
            index=self.pattern_names
        )

    def _create_visualizations(
        self,
        ratio_df: pd.DataFrame,
        p_value_df: pd.DataFrame,
        pattern_names: List[str],
        n_gen: int
    ):
        """Create visualization plots for PA ratio analysis results.

        Args:
            ratio_df: DataFrame containing PA ratios with patterns as index and
                cell types as columns.
            p_value_df: DataFrame containing corrected p-values with same structure
                as ratio_df.
            pattern_names: List of pattern names to include in visualizations.
            n_gen: Number of generations/matches used in the analysis.
                Used for output file naming.

        Notes:
            Saves the generated plots to the output directory with filename
            format 'PA_ratio_k{n_gen}.pdf'.
        """
        # Create bar plot
        self._create_pa_ratio_bar_plot(
            ratio_df,
            p_value_df,
            f'PA Ratio',
            self.output_dir / f'PA_ratio_k{n_gen}.pdf'
        )

    def _create_pa_ratio_bar_plot(
        self,
        data: pd.DataFrame,
        p_value_df: pd.DataFrame,
        title: str,
        output_path: Path,
        alpha: float = 0.05
    ):
        """Create a bar plot of PA ratios with significance markers.

        Args:
            data: DataFrame containing PA ratios with patterns as index and
                cell types as columns.
            p_value_df: DataFrame containing p-values with same structure as data.
            title: Title for the plot.
            output_path: Path object specifying where to save the plot.
            alpha: Significance threshold for marking significant results.
                Defaults to 0.05.

        Notes:
            - Melts the input data for plotting with patterns on x-axis and
              different cell types as grouped bars
            - Adds significance markers (*) for results where p < alpha
            - Customizes plot appearance including colors, labels, and legend
            - Saves the plot to the specified output path
        """
        pattern_names = data.index.tolist()

        # Prepare data for plotting
        melted_data = data.reset_index().melt(
            id_vars='index',
            var_name='Cell Type',
            value_name='PA Ratio'
        )
        melted_data = melted_data.rename(columns={'index': 'Pattern'})

        # Create and customize plot
        fig, ax = self._setup_bar_plot(melted_data)

        # Plot bars and add significance markers
        bar_positions = self._plot_bars(ax, melted_data, pattern_names, p_value_df, alpha)

        # Customize plot appearance
        self._customize_bar_plot(
            ax,
            title,
            melted_data,
            pattern_names,
            bar_positions
        )

        # Save plot
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def _setup_bar_plot(
        self,
        melted_data: pd.DataFrame
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Set up the basic figure and axes for the PA ratio bar plot.

        Args:
            melted_data: DataFrame in long format containing Pattern, Cell Type,
                and PA Ratio columns.

        Returns:
            tuple: A tuple containing:
                - fig (plt.Figure): The created figure object
                - ax (plt.Axes): The axes object for plotting

        Notes:
            Creates a figure with dimensions 16x7 inches.
        """
        fig, ax = plt.subplots(figsize=(16, 7))
        return fig, ax

    def _plot_bars(
        self,
        ax: plt.Axes,
        melted_data: pd.DataFrame,
        patterns: List[str],
        p_value_df: pd.DataFrame,
        alpha: float
    ) -> Dict[str, np.ndarray]:
        """Plot grouped bar chart of PA ratios with significance markers.

        Args:
            ax: Matplotlib axes object to plot on.
            melted_data: DataFrame in long format containing Pattern, Cell Type,
                and PA Ratio columns.
            patterns: List of pattern names to include in the plot.
            p_value_df: DataFrame containing p-values with patterns as index and
                cell types as columns.
            alpha: Significance threshold for marking significant results.

        Returns:
            dict: Dictionary mapping pattern names to arrays of bar positions
                on the x-axis.

        Notes:
            - Uses a bar width of 0.15 and gap width of 0.02 between bars
            - Groups bars by cell type
            - Adds black edges to bars for better visibility
            - Skips significance markers for 'All' pattern
        """
        bar_width = 0.15
        gap_width = 0.02
        cell_types = melted_data['Cell Type'].unique()
        group_width = len(patterns) * (bar_width + gap_width) - gap_width
        group_positions = np.arange(len(cell_types))
        bar_positions_dict = {}

        for i, pattern in enumerate(patterns):
            pattern_data = melted_data[melted_data['Pattern'] == pattern]
            bar_positions = group_positions + i * (bar_width + gap_width) - group_width / 2 + bar_width / 2

            ax.bar(
                bar_positions,
                pattern_data['PA Ratio'],
                width=bar_width,
                label=pattern,
                edgecolor='black',
                linewidth=1
            )

            bar_positions_dict[pattern] = bar_positions

            # Add significance markers
            if pattern != 'All':
                self._add_significance_markers(
                    ax,
                    pattern,
                    cell_types,
                    bar_positions,
                    pattern_data,
                    p_value_df,
                    alpha
                )

        return bar_positions_dict

    def _add_significance_markers(
        self,
        ax: plt.Axes,
        pattern: str,
        cell_types: np.ndarray,
        bar_positions: np.ndarray,
        pattern_data: pd.DataFrame,
        p_value_df: pd.DataFrame,
        alpha: float
    ):
        """Add significance markers above bars based on p-values.

        Args:
            ax: Matplotlib axes object to add markers to.
            pattern: Name of the current pattern.
            cell_types: Array of cell type names.
            bar_positions: Array of x-axis positions for the bars.
            pattern_data: DataFrame containing PA ratios for the current pattern.
            p_value_df: DataFrame containing p-values with patterns as index and
                cell types as columns.
            alpha: Significance threshold for marking significant results.

        Notes:
            Adds markers based on p-value thresholds:
            - '***': p < 0.001
            - '**': p < 0.01
            - '*': p < 0.05
            Only adds markers for significant results (p < alpha)
        """
        for j, cell_type in enumerate(cell_types):
            p_value = p_value_df.loc[pattern, cell_type]

            if not np.isnan(p_value) and p_value < alpha:
                current_height = pattern_data[pattern_data['Cell Type'] == cell_type]['PA Ratio'].values[0]

                marker = (
                    '***' if p_value < 0.001 else
                    '**' if p_value < 0.01 else
                    '*'
                )

                y_pos = current_height + current_height * 0.05
                ax.text(
                    bar_positions[j],
                    y_pos,
                    marker,
                    ha='center',
                    va='bottom'
                )

    def _customize_bar_plot(
        self,
        ax: plt.Axes,
        title: str,
        melted_data: pd.DataFrame,
        patterns: List[str],
        bar_positions: Dict[str, np.ndarray]
    ):
        """Customize the appearance of the PA ratio bar plot.

        Args:
            ax: Matplotlib axes object to customize.
            title: Title for the plot.
            melted_data: DataFrame in long format containing Pattern, Cell Type,
                and PA Ratio columns.
            patterns: List of pattern names included in the plot.
            bar_positions: Dictionary mapping pattern names to arrays of bar positions
                on the x-axis.

        Notes:
            Customizes the following plot elements:
            - Adds labels and title with specified font sizes
            - Sets x-axis ticks and rotated labels for cell types
            - Adjusts y-axis limit to 120% of maximum PA ratio
            - Adds legend with site patterns
            - Includes a text box explaining significance markers
            - Applies tight layout for better spacing

        Style Details:
            - Font size: 12 for axis labels
            - X-tick labels: 45-degree rotation, right-aligned
            - Legend: Positioned outside plot on the right
            - Significance legend: Positioned at (1.05, 0.5) in axes coordinates
        """
        # Set labels and title
        ax.set_xlabel('Cell Types', fontsize=12)
        ax.set_ylabel('PA Ratio', fontsize=12)
        ax.set_title(title)

        # Set x-ticks
        cell_types = melted_data['Cell Type'].unique()
        group_positions = np.arange(len(cell_types))
        ax.set_xticks(group_positions)
        ax.set_xticklabels(cell_types, rotation=45, ha='right')

        # Set y-axis limit
        max_ratio = melted_data['PA Ratio'].max()
        ax.set_ylim(0, max_ratio * 1.15)

        # Add legend
        ax.legend(
            title='Site Patterns',
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )

        # Add significance level explanation
        ax.text(
            1.05, 0.5,
            '* p < 0.05\n** p < 0.01\n*** p < 0.001\nCompared to All',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )

        plt.tight_layout()

    def _save_results(
        self,
        ratio_df: pd.DataFrame,
        p_value_df: pd.DataFrame,
        num_df: pd.DataFrame,
        n_gen: int
    ):
        """Save PA ratio analysis results to CSV files.

        Args:
            ratio_df: DataFrame containing PA ratios with patterns as index and
                cell types as columns.
            p_value_df: DataFrame containing corrected p-values with same structure
                as ratio_df.
            num_df: DataFrame containing cell counts with same structure as ratio_df.
            n_gen: Number of generations/matches used in the analysis.

        Notes:
            Saves three CSV files in the output directory:
            - PA_ratio_all_patients_k{n_gen}.csv: Contains PA ratios
            - p_value_all_patients_k{n_gen}.csv: Contains corrected p-values
            - num_cells_all_patients_k{n_gen}.csv: Contains cell counts

            All files maintain the same structure with patterns as rows and
            cell types as columns.
        """
        ratio_df.to_csv(self.output_dir / f'PA_ratio_all_patients_k{n_gen}.csv')
        p_value_df.to_csv(self.output_dir / f'p_value_all_patients_k{n_gen}.csv')
        num_df.to_csv(self.output_dir / f'num_cells_all_patients_k{n_gen}.csv')
