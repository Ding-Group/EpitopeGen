import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from .config import GENES_OF_INTEREST,GENE_GROUPS

class DEGAnalyzer:
    """A class for performing differential expression gene analysis between PA and NA cells.

    Analyzes differential gene expression between phenotype-associated (PA) and
    not-associated (NA) cells, with support for custom gene groups and site patterns.

    Attributes:
        genes_of_interest: List of genes to analyze.
        gene_groups: Dictionary mapping group names to lists of genes.
        patterns_list: List of lists containing site patterns to analyze.
        pattern_names: Names corresponding to site patterns.
        output_dir: Path object for the results directory.
        top_k: Number of top matches to consider for PA cells.
    """
    def __init__(
        self,
        genes_of_interest: List[str] = GENES_OF_INTEREST,
        gene_groups: Dict[str, List[str]] = GENE_GROUPS,
        patterns_list: Optional[List[List[str]]] = None,
        pattern_names: Optional[List[str]] = None,
        output_dir: str = "results",
        top_k: int = 1
    ):
        """Initializes the DEG analyzer with specified parameters.

        Args:
            genes_of_interest: List of genes to analyze in the differential
                expression analysis.
            gene_groups: Dictionary mapping group names to lists of genes for
                grouped analysis.
            patterns_list: List of lists containing site patterns to analyze.
                Optional for pattern-specific analysis.
            pattern_names: Names corresponding to site patterns. Required if
                patterns_list is provided.
            output_dir: Directory path to save analysis results (default: "results").
            top_k: Number of top matches to consider for PA cells (default: 1).

        Raises:
            ValueError: If input parameters are invalid or inconsistent.
        """
        self.genes_of_interest = genes_of_interest
        self.gene_groups = gene_groups
        self.patterns_list = patterns_list
        self.pattern_names = pattern_names
        self.output_dir = Path(output_dir)
        self.top_k = top_k

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self):
        """Validates the initialization input parameters.

        Checks for:
            - Non-empty genes_of_interest list
            - Non-empty gene_groups dictionary
            - Matching lengths of patterns_list and pattern_names if provided

        Raises:
            ValueError: If any validation check fails.
        """
        if not self.genes_of_interest:
            raise ValueError("genes_of_interest cannot be empty")

        if not self.gene_groups:
            raise ValueError("gene_groups cannot be empty")

        if self.patterns_list is not None:
            if not self.pattern_names:
                raise ValueError("pattern_names must be provided when patterns_list is specified")
            if len(self.patterns_list) != len(self.pattern_names):
                raise ValueError("Length of patterns_list must match pattern_names")

    def prepare_data(self, adata: sc.AnnData) -> sc.AnnData:
        """Preprocesses the input gene expression data.

        Performs standard single-cell preprocessing steps including total count
        normalization and log transformation.

        Args:
            adata: AnnData object containing gene expression data.

        Returns:
            sc.AnnData: Preprocessed copy of the input data with:
                - Total counts normalized to 1e4
                - Log1p transformed expression values

        Note:
            Creates a copy of the input data to preserve the original.
        """
        adata = adata.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        return adata

    def _create_pa_mask(self, adata: sc.AnnData) -> pd.Series:
        """Creates a boolean mask identifying phenotype-associated (PA) cells.

        Identifies PA cells by checking match columns up to top_k, where a cell
        is considered PA if it has a match in any of the considered positions.

        Args:
            adata: AnnData object containing match information in obs columns
                (match_0 through match_{top_k-1}).

        Returns:
            pd.Series: Boolean mask with True for PA cells and False for NA cells,
                indexed by cell names.

        Note:
            Looks for match columns named 'match_0' through 'match_{top_k-1}'
            in adata.obs.
        """
        pa_mask = pd.Series(False, index=adata.obs.index)
        for k in range(self.top_k):
            match_col = f'match_{k}'
            if match_col in adata.obs.columns:
                pa_mask |= (adata.obs[match_col] == 1)
        return pa_mask

    def perform_deg_analysis(self, adata: sc.AnnData) -> Tuple[Dict, int, int]:
        """Performs differential expression analysis between PA and NA cells.

        Conducts Wilcoxon rank-sum test to identify differentially expressed genes
        between phenotype-associated (PA) and non-associated (NA) cells.

        Args:
            adata: AnnData object containing gene expression data and match information
                in obs columns.

        Returns:
            tuple: A tuple containing:
                - dict: Results dictionary mapping each gene to its statistics:
                    - logfoldchange: Log fold change between PA and NA cells
                    - pval: Raw p-value from Wilcoxon test
                    - pval_adj: Adjusted p-value for multiple testing
                    - pct_pa: Percentage of PA cells expressing the gene
                    - pct_na: Percentage of NA cells expressing the gene
                - int: Number of PA cells identified
                - int: Total number of cells analyzed

        Note:
            - Uses scanpy's rank_genes_groups with Wilcoxon test
            - Returns empty results with appropriate structure if analysis fails
            - Handles missing genes and failed computations gracefully
        """
        # Initialize groups
        adata.obs['comparison_group'] = 'NA'
        pa_mask = self._create_pa_mask(adata)
        adata.obs.loc[pa_mask, 'comparison_group'] = 'PA'

        # Calculate statistics
        n_pa = sum(adata.obs['comparison_group'] == 'PA')
        n_total = len(adata.obs)
        print(f"PA cells (top {self.top_k} matches): {n_pa} ({n_pa/n_total*100:.2f}%) out of {n_total} total cells")

        try:
            # Perform DEG analysis
            sc.tl.rank_genes_groups(
                adata,
                groupby='comparison_group',
                groups=['PA'],
                reference='NA',
                method='wilcoxon',
                pts=True
            )

            # Extract results using the correct API
            try:
                de_results = pd.DataFrame(
                    {group + '_' + key: adata.uns['rank_genes_groups'][key][group]
                    for group in ['PA']
                    for key in ['names', 'logfoldchanges', 'pvals', 'pvals_adj', 'pts']})
            except Exception as e:
                print(f"Warning: Could not get rank genes groups results: {str(e)}")
                de_results = pd.DataFrame()  # Empty DataFrame as fallback

            results = {}
            for gene in self.genes_of_interest:
                try:
                    if gene in adata.var_names:
                        gene_results = de_results[de_results['PA_names'] == gene]
                        if len(gene_results) > 0:
                            results[gene] = {
                                'logfoldchange': gene_results['PA_logfoldchanges'].iloc[0],
                                'pval': gene_results['PA_pvals'].iloc[0],
                                'pval_adj': gene_results['PA_pvals_adj'].iloc[0],
                                'pct_pa': gene_results['PA_pts'].iloc[0] if 'PA_pts' in gene_results.columns else None,
                                'pct_na': None  # This would need additional calculation if needed
                            }
                        else:
                            print(f"Warning: No results found for gene {gene}")
                            results[gene] = self._get_empty_result()
                    else:
                        print(f"Warning: Gene {gene} not found in the dataset")
                        results[gene] = self._get_empty_result()
                except Exception as e:
                    print(f"Warning: Error processing gene {gene}: {str(e)}")
                    results[gene] = self._get_empty_result()

            return results, n_pa, n_total

        except Exception as e:
            print(f"Error in differential expression analysis: {str(e)}")
            return {gene: self._get_empty_result() for gene in self.genes_of_interest}, 0, 0

    def _get_empty_result(self) -> Dict:
        """Creates an empty result dictionary for missing or failed gene analyses.

        Returns:
            dict: Dictionary with empty/null values for all result fields:
                - logfoldchange: np.nan
                - pval: np.nan
                - pval_adj: np.nan
                - pct_pa: None
                - pct_na: None

        Note:
            Used as a fallback when gene analysis fails or gene is not found
            in the dataset.
        """
        return {
            'logfoldchange': np.nan,
            'pval': np.nan,
            'pval_adj': np.nan,
            'pct_pa': None,
            'pct_na': None
        }

    def create_visualization(
        self,
        results: Dict,
        pattern_names_with_stats: Optional[List[str]] = None,
        is_heatmap: bool = False
    ):
        """Creates visualization of differential expression analysis results.

        Generates either a heatmap or barplot visualization of the DEG analysis
        results, depending on the specified parameters.

        Args:
            results: Dictionary containing differential expression results for
                each gene.
            pattern_names_with_stats: Optional list of pattern names with their
                statistics for labeling (default: None).
            is_heatmap: Whether to create a heatmap (True) or barplot (False)
                visualization (default: False).

        Note:
            - For heatmaps, uses pattern_names_with_stats for row labels
            - For barplots, shows logfoldchange values with significance markers
            - Saves visualization to the specified output directory
        """
        if is_heatmap:
            self._create_heatmap_visualization(results, pattern_names_with_stats)
        else:
            self._create_barplot_visualization(results)

    def _create_barplot_visualization(self, results: Dict):
        """Creates a grouped bar plot visualization of differential expression results.

        Generates a bar plot showing log fold changes for each gene, grouped by their
        functional categories, with significance markers.

        Args:
            results: Dictionary mapping genes to their differential expression results,
                containing:
                - logfoldchange: Log2 fold change between PA and NA cells
                - pval_adj: Adjusted p-value for significance testing

        Note:
            - Bars are colored red for positive and blue for negative fold changes
            - Significance levels are indicated with stars above/below bars
            - Groups are labeled on the x-axis with vertical text
            - Plot is saved as 'deg_grouped_topk_{k}.pdf' in the output directory
            - Y-axis is limited to [-2, 2] for better visualization
        """
        plt.figure(figsize=(15, 10))
        plt.ylim(-2, 2)

        # Calculate positions for grouped bars
        group_positions = []
        current_pos = 0

        for group_name, genes in self.gene_groups.items():
            group_genes = [g for g in genes if g in results]
            if group_genes:
                group_positions.append((group_name, current_pos, current_pos + len(group_genes)))
                current_pos += len(group_genes) + 1.5

        bar_positions = []
        bar_labels = []
        y_offset = 0.1

        for group_name, start_pos, end_pos in group_positions:
            group_genes = [g for g in self.gene_groups[group_name] if g in results]
            positions = np.arange(start_pos, start_pos + len(group_genes))

            # Plot bars
            for pos, gene in zip(positions, group_genes):
                lfc = results[gene]['logfoldchange']
                color = 'red' if lfc >= 0 else 'blue'
                plt.bar(pos, lfc, color=color, alpha=0.7)

                # Add significance markers
                stars = self._get_significance_stars(results[gene]['pval_adj'])
                text_y = min(lfc + y_offset, 1.9) if lfc >= 0 else max(lfc - y_offset, -1.9)
                plt.text(pos, text_y, stars, ha='center', va='bottom' if lfc >= 0 else 'top')

            bar_positions.extend(positions)
            bar_labels.extend(group_genes)

            # Add group labels
            group_center = np.mean(positions)
            plt.text(group_center, -2.2, group_name, ha='center', va='top', rotation=90)

        # Customize plot
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        plt.title(f"Differential Gene Expression (Log2 Fold Change PA vs NA)\nAll Cell Types (Top {self.top_k} matches)",
                 fontsize=12, pad=20)
        plt.xlabel("Genes", fontsize=10)
        plt.ylabel("Log2 Fold Change", fontsize=10)
        plt.xticks(bar_positions, bar_labels, rotation=90, ha='center')

        # Add legends
        self._create_plot_legends()

        # Save plot
        plt.savefig(self.output_dir / f"deg_grouped_topk_{self.top_k}.pdf",
                    format='pdf', bbox_inches='tight', dpi=300)
        plt.close()

    def _create_heatmap_visualization(self, results: Dict, pattern_names_with_stats: List[str]):
        """Creates a heatmap visualization of differential expression across patterns.

        Generates a heatmap showing log fold changes for each gene across different
        expansion patterns, with genes grouped by functional categories and sorted
        by average fold change within groups.

        Args:
            results: Nested dictionary structure:
                {pattern: {gene: {logfoldchange: float, pval_adj: float}}}
                containing differential expression results for each pattern and gene.
            pattern_names_with_stats: List of pattern names with additional statistics
                for column labels.

        Note:
            - Uses RdBu_r colormap centered at 0
            - Includes significance markers for adjusted p-values
            - Groups are separated by horizontal lines
            - Saves both PDF and PNG versions
            - Numerical results are saved separately
            - Heatmap cell values are formatted to 2 decimal places
        """
        # Initialize lists to store genes and track group boundaries
        all_genes = []
        group_boundaries = []
        current_position = 0

        # Prepare data for heatmap
        fold_changes = {gene: [] for gene in results[self.pattern_names[0]].keys()}
        p_values = {gene: [] for gene in results[self.pattern_names[0]].keys()}

        for pattern in self.pattern_names:
            for gene in fold_changes.keys():
                fold_changes[gene].append(results[pattern][gene]['logfoldchange'])
                p_values[gene].append(results[pattern][gene]['pval_adj'])

        # Collect valid genes from each group and track boundaries
        for group_name, genes in self.gene_groups.items():
            valid_genes = [gene for gene in genes if gene in fold_changes]
            if valid_genes:
                avg_fold_changes = {gene: np.nanmean(fold_changes[gene]) for gene in valid_genes}
                sorted_group_genes = sorted(valid_genes, key=lambda g: avg_fold_changes[g], reverse=True)

                all_genes.extend(sorted_group_genes)
                current_position += len(sorted_group_genes)
                group_boundaries.append((group_name, current_position))

        # Create heatmap data array
        heatmap_data = np.array([fold_changes[gene] for gene in all_genes])

        # Create and customize heatmap
        plt.figure(figsize=(12, len(all_genes) * 0.4 + 1))
        sns.heatmap(heatmap_data,
                    annot=True,     # Show values in cells
                    fmt=".2f",      # Format for cell values
                    cmap="RdBu_r",  # Color map (red-blue)
                    center=0,
                    vmin=-2,        # Clip minimum value for color scaling
                    vmax=2,         # Clip maximum value for color scaling
                    xticklabels=pattern_names_with_stats,
                    yticklabels=all_genes,
                    cbar_kws={'label': 'Log2 Fold Change (PA/NA)'})

        # Add significance markers
        self._add_significance_markers(all_genes, self.pattern_names, p_values)

        # Add group labels and boundaries
        self._add_group_boundaries(group_boundaries, all_genes)

        plt.title(f'Gene Expression Differences (PA vs NA) Across Expansion Patterns\n(Top {self.top_k} matches)')
        plt.xlabel('Expansion Pattern')
        plt.ylabel('Genes')

        # Save plots
        plt.tight_layout()
        plt.savefig(self.output_dir / f"gene_expression_heatmap_k{self.top_k}.pdf",
                    format='pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / f"gene_expression_heatmap_k{self.top_k}.png",
                    format='png', dpi=300, bbox_inches='tight')
        plt.close()

        # Save numerical results
        self._save_numerical_results(all_genes, group_boundaries, fold_changes, p_values, self.pattern_names)

    def _get_significance_stars(self, pvalue: float) -> str:
        """Converts p-values to significance star notation.

        Args:
            pvalue: Adjusted p-value from statistical test.

        Returns:
            str: Significance stars:
                - '***' for p ≤ 0.001
                - '**' for p ≤ 0.01
                - '*' for p ≤ 0.05
                - '' (empty string) for p > 0.05
        """
        if pvalue <= 0.001: return '***'
        elif pvalue <= 0.01: return '**'
        elif pvalue <= 0.05: return '*'
        return ''

    def _create_plot_legends(self):
        """Creates and positions legends for significance levels and color coding.

        Adds two legends to the current plot:
            1. Color legend showing up/down regulation in PA cells
            2. Significance level legend showing p-value thresholds

        Note:
            - Positions legends outside the main plot area
            - Uses alpha=0.7 for color patches
            - Adjusts subplot parameters to prevent legend overlap
        """
        significance_elements = [
            plt.Text(0, 0, '*** p ≤ 0.001'),
            plt.Text(0, 0, '** p ≤ 0.01'),
            plt.Text(0, 0, '* p ≤ 0.05'),
            plt.Text(0, 0, 'ns p > 0.05')
        ]
        color_elements = [
            plt.Rectangle((0,0),1,1, fc='red', alpha=0.7, label='Upregulated in PA'),
            plt.Rectangle((0,0),1,1, fc='blue', alpha=0.7, label='Downregulated in PA')
        ]

        plt.legend(handles=color_elements, bbox_to_anchor=(1.05, 1),
                  loc='upper left', title='Expression Change')
        sig_legend = plt.legend(handles=significance_elements, bbox_to_anchor=(1.05, 0.6),
                               loc='center left', title='Adjusted Significance Levels')
        plt.gca().add_artist(sig_legend)
        plt.subplots_adjust(bottom=0.2)

    def _add_significance_markers(self, all_genes: List[str], pattern_names: List[str], p_values: Dict):
        """Adds significance markers to heatmap cells.

        Overlays significance stars on heatmap cells based on adjusted p-values.

        Args:
            all_genes: List of gene names in order of appearance on heatmap.
            pattern_names: List of pattern names defining heatmap columns.
            p_values: Dictionary mapping genes to lists of p-values, one per pattern.

        Note:
            Uses standard significance levels:
            - '***' for p < 0.001
            - '**' for p < 0.01
            - '*' for p < 0.05
        """
        for i, gene in enumerate(all_genes):
            for j in range(len(pattern_names)):
                if p_values[gene][j] < 0.001:
                    plt.text(j + 0.7, i + 0.5, '***', ha='center', va='center')
                elif p_values[gene][j] < 0.01:
                    plt.text(j + 0.7, i + 0.5, '**', ha='center', va='center')
                elif p_values[gene][j] < 0.05:
                    plt.text(j + 0.7, i + 0.5, '*', ha='center', va='center')

    def _add_group_boundaries(self, group_boundaries: List[Tuple[str, int]], all_genes: List[str]):
        """Adds group labels and visual boundaries to heatmap.

        Adds group names on the left side of the heatmap and horizontal white
        lines between groups.

        Args:
            group_boundaries: List of tuples containing (group_name, ending_position)
                for each gene group.
            all_genes: List of all gene names to determine total heatmap height.

        Note:
            - Group names are right-aligned and bold
            - White horizontal lines separate groups
            - Group labels are centered vertically within their group
        """
        prev_pos = 0
        for group_name, end_pos in group_boundaries:
            middle_pos = prev_pos + (end_pos - prev_pos)/2
            plt.text(-0.5, middle_pos, group_name,
                    ha='right', va='center',
                    fontweight='bold')
            if end_pos < len(all_genes):
                plt.axhline(y=end_pos, color='white', linewidth=2)
            prev_pos = end_pos

    def _save_numerical_results(
        self,
        all_genes: List[str],
        group_boundaries: List[Tuple[str, int]],
        fold_changes: Dict,
        p_values: Dict,
        pattern_names: List[str]
    ):
        """Saves numerical results of differential expression analysis to CSV.

        Creates a comprehensive CSV file containing fold changes and p-values
        for all genes across all patterns, including group assignments.

        Args:
            all_genes: Ordered list of genes included in the analysis.
            group_boundaries: List of tuples containing (group_name, ending_position)
                defining gene group boundaries.
            fold_changes: Dictionary mapping genes to their fold change values
                across patterns.
            p_values: Dictionary mapping genes to their p-values across patterns.
            pattern_names: List of pattern names used in the analysis.

        Note:
            Saves file as 'expression_analysis_results_k{top_k}.csv' in the
            output directory with columns:
            - Gene: Gene name
            - Group: Functional group assignment
            - {pattern}_fold_change: Log2 fold change for each pattern
            - {pattern}_pvalue: Adjusted p-value for each pattern
        """
        results_df = pd.DataFrame({
            'Gene': all_genes,
            'Group': [next(group_name for group_name, end_pos in group_boundaries if end_pos > i)
                     for i in range(len(all_genes))],
            **{f"{pattern}_fold_change": [fold_changes[gene][i] for gene in all_genes]
               for i, pattern in enumerate(pattern_names)},
            **{f"{pattern}_pvalue": [p_values[gene][i] for gene in all_genes]
               for i, pattern in enumerate(pattern_names)}
        })
        results_df.to_csv(self.output_dir / f"expression_analysis_results_k{self.top_k}.csv", index=False)

    def analyze(
        self,
        adata: sc.AnnData,
        analyze_patterns: bool = False
    ) -> pd.DataFrame:
        """Performs comprehensive differential expression analysis.

        Main entry point for running the differential expression analysis,
        with options for pattern-specific or global analysis.

        Args:
            adata: AnnData object containing gene expression data and cell
                annotations.
            analyze_patterns: If True, performs separate analyses for each pattern
                in patterns_list. If False, analyzes all cells together
                (default: False).

        Returns:
            pd.DataFrame: Analysis results with genes as index and statistical
                measures as columns. Format depends on analysis type:
                - Global analysis: logfoldchange, pval, pval_adj per gene
                - Pattern analysis: above statistics for each pattern

        Note:
            Automatically preprocesses data before analysis, including
            normalization and log transformation.
        """
        # Preprocess data
        adata = self.prepare_data(adata)

        if analyze_patterns and self.patterns_list:
            return self._analyze_patterns(adata)
        else:
            return self._analyze_all(adata)

    def _analyze_patterns(self, adata: sc.AnnData) -> pd.DataFrame:
        """Performs separate differential expression analyses for each pattern.

        Analyzes differential expression between PA and NA cells for each pattern
        in patterns_list, calculating statistics and storing results.

        Args:
            adata: AnnData object containing preprocessed gene expression data
                and a 'pattern' column in obs containing pattern assignments.

        Returns:
            pd.DataFrame: Combined analysis results with columns:
                {pattern_name}_{stat} for each pattern and statistic, where:
                - pattern_name: Name of each analyzed pattern
                - stat: One of [logfoldchange, pval, pval_adj, pct_pa, pct_na]

        Note:
            - Creates visualization of results across patterns
            - Maintains statistics about PA cell percentages per pattern
            - Filters data to include only cells belonging to each pattern
        """
        pattern_results = {}
        pattern_stats = {}

        for pattern, pattern_name in zip(self.patterns_list, self.pattern_names):
            # Filter data for current pattern
            pattern_adata = adata[adata.obs['pattern'].isin(pattern)].copy()

            # Perform analysis
            results, n_pa, n_total = self.perform_deg_analysis(pattern_adata)
            pattern_results[pattern_name] = results
            pattern_stats[pattern_name] = {
                'total': n_total,
                'pa': n_pa,
                'percent_pa': (n_pa/n_total*100) if n_total > 0 else 0
            }

        # Create visualization with statistics
        pattern_names_with_stats = [
            f"{pattern}\n(Total N={pattern_stats[pattern]['total']:,}\nPA N={pattern_stats[pattern]['pa']:,}, {pattern_stats[pattern]['percent_pa']:.1f}%)"
            for pattern in self.pattern_names
        ]

        self.create_visualization(
            pattern_results,
            pattern_names_with_stats=pattern_names_with_stats,
            is_heatmap=True
        )

        # Prepare results DataFrame
        return self._prepare_pattern_results_df(pattern_results)

    def _analyze_all(self, adata: sc.AnnData) -> pd.DataFrame:
        """Performs global differential expression analysis across all cells.

        Analyzes differential expression between PA and NA cells without
        pattern stratification, creating visualizations and saving results.

        Args:
            adata: AnnData object containing preprocessed gene expression data
                and cell annotations.

        Returns:
            pd.DataFrame: Analysis results with genes as index and columns for:
                - logfoldchange: Log2 fold change between PA and NA cells
                - pval: Raw p-value
                - pval_adj: Adjusted p-value
                - pct_pa: Percentage of PA cells expressing the gene
                - pct_na: Percentage of NA cells expressing the gene

        Note:
            Creates bar plot visualization of results automatically and saves
            to output directory.
        """
        results, _, _ = self.perform_deg_analysis(adata)

        self.create_visualization(
            results,
            is_heatmap=False
        )

        return pd.DataFrame.from_dict(results, orient='index')

    def _prepare_pattern_results_df(self, pattern_results: Dict) -> pd.DataFrame:
        """Converts pattern-specific results into a combined DataFrame.

        Processes the nested dictionary of pattern results into a wide-format
        DataFrame with pattern-specific columns for each statistical measure.

        Args:
            pattern_results: Dictionary mapping pattern names to their respective
                differential expression results dictionaries. Structure:
                {pattern_name: {gene: {stat_name: value}}}

        Returns:
            pd.DataFrame: Wide-format DataFrame with:
                - Index: Gene names
                - Columns: {pattern}_{stat} for each pattern and statistical measure
                - Values: Statistical results for each gene-pattern combination

        Note:
            Column names are prefixed with pattern names to distinguish
            statistics across patterns.
        """
        results_df = pd.DataFrame()
        for pattern in self.pattern_names:
            pattern_df = pd.DataFrame.from_dict(pattern_results[pattern], orient='index')
            pattern_df = pattern_df.add_prefix(f'{pattern}_')
            if results_df.empty:
                results_df = pattern_df
            else:
                results_df = pd.concat([results_df, pattern_df], axis=1)
        return results_df
