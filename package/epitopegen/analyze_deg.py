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
    """Class for performing differential expression gene analysis between PA and NA cells."""

    def __init__(
        self,
        genes_of_interest: List[str] = GENES_OF_INTEREST,
        gene_groups: Dict[str, List[str]] = GENE_GROUPS,
        patterns_list: Optional[List[List[str]]] = None,
        pattern_names: Optional[List[str]] = None,
        output_dir: str = "results",
        top_k: int = 1
    ):
        """Initialize DEG analyzer.

        Args:
            genes_of_interest: List of genes to analyze
            gene_groups: Dictionary mapping group names to lists of genes
            patterns_list: List of lists containing site patterns to analyze
            pattern_names: Names corresponding to site patterns
            output_dir: Directory to save results (default: "results")
            top_k: Number of top matches to consider for PA cells (default: 1)
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
        """Validate input parameters."""
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
        """Preprocess the input data.

        Args:
            adata: AnnData object containing gene expression data

        Returns:
            Preprocessed AnnData object
        """
        adata = adata.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        return adata

    def _create_pa_mask(self, adata: sc.AnnData) -> pd.Series:
        """Create mask for PA cells considering top K matches."""
        pa_mask = pd.Series(False, index=adata.obs.index)
        for k in range(self.top_k):
            match_col = f'match_{k}'
            if match_col in adata.obs.columns:
                pa_mask |= (adata.obs[match_col] == 1)
        return pa_mask

    def perform_deg_analysis(self, adata: sc.AnnData) -> Tuple[Dict, int, int]:
        """Perform differential expression analysis between PA and NA cells.

        Args:
            adata: AnnData object containing gene expression data

        Returns:
            Tuple containing:
                - Dictionary of DEG results
                - Number of PA cells
                - Total number of cells
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
        """Get empty result dictionary for missing genes."""
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
        """Create visualization of differential expression results."""
        if is_heatmap:
            self._create_heatmap_visualization(results, pattern_names_with_stats)
        else:
            self._create_barplot_visualization(results)

    def _create_barplot_visualization(self, results: Dict):
        """Create grouped bar plot visualization of DEG results."""
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
        """Create heatmap visualization of DEG results across patterns."""
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
                    annot=True,
                    fmt=".2f",
                    cmap="RdBu_r",
                    center=0,
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
        """Get significance stars based on p-value."""
        if pvalue <= 0.001: return '***'
        elif pvalue <= 0.01: return '**'
        elif pvalue <= 0.05: return '*'
        return ''

    def _create_plot_legends(self):
        """Create and add legends to the current plot."""
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
        """Add significance markers to heatmap."""
        for i, gene in enumerate(all_genes):
            for j in range(len(pattern_names)):
                if p_values[gene][j] < 0.001:
                    plt.text(j + 0.7, i + 0.5, '***', ha='center', va='center')
                elif p_values[gene][j] < 0.01:
                    plt.text(j + 0.7, i + 0.5, '**', ha='center', va='center')
                elif p_values[gene][j] < 0.05:
                    plt.text(j + 0.7, i + 0.5, '*', ha='center', va='center')

    def _add_group_boundaries(self, group_boundaries: List[Tuple[str, int]], all_genes: List[str]):
        """Add group labels and boundaries to heatmap."""
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
        """Save numerical results to CSV."""
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
        """Main method to analyze gene expression differences.

        Args:
            adata: AnnData object containing gene expression data
            analyze_patterns: Whether to analyze expression patterns separately

        Returns:
            DataFrame containing analysis results
        """
        # Preprocess data
        adata = self.prepare_data(adata)

        if analyze_patterns and self.patterns_list:
            return self._analyze_patterns(adata)
        else:
            return self._analyze_all(adata)

    def _analyze_patterns(self, adata: sc.AnnData) -> pd.DataFrame:
        """Analyze expression patterns separately."""
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
        """Analyze all cells together."""
        results, _, _ = self.perform_deg_analysis(adata)

        self.create_visualization(
            results,
            is_heatmap=False
        )

        return pd.DataFrame.from_dict(results, orient='index')

    def _prepare_pattern_results_df(self, pattern_results: Dict) -> pd.DataFrame:
        """Prepare results DataFrame from pattern analysis."""
        results_df = pd.DataFrame()
        for pattern in self.pattern_names:
            pattern_df = pd.DataFrame.from_dict(pattern_results[pattern], orient='index')
            pattern_df = pattern_df.add_prefix(f'{pattern}_')
            if results_df.empty:
                results_df = pattern_df
            else:
                results_df = pd.concat([results_df, pattern_df], axis=1)
        return results_df
