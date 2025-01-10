# Standard library imports
import math
import multiprocessing as mp
import os
from itertools import combinations
from multiprocessing import Pool, cpu_count
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from Bio.Align import substitution_matrices
from scipy import stats
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.multitest import multipletests
from umap import UMAP

# Local imports
from covid19_su.utils import *


def analysis_wrapper(data_dir, pred_csv, gex_cache, epi_db_path, obs_cache=None, outdir=None):
    # A wrapper function to run series of annotations
    adata = read_all_data(data_dir, gex_cache=gex_cache, obs_cache=obs_cache, use_multiprocessing=False, filtering=False)

    PA_ratio_analysis(adata.copy(), outdir=f"{outdir}/PA_ratio", swap_axes=True)
    for k in [8]:
        expression_level_analysis2(adata.copy(), outdir=f"{outdir}/gex2", k=k)
    for k in [8, 16, 32]:
        expression_level_analysis(adata.copy(), outdir=f"{outdir}/gex", k=k)
    for k in [8, 16, 32]:
        expression_level_analysis_celltype(adata.copy(), outdir=f"{outdir}/gex_celltype", k=k)
    for k in [7,8,9,10]:
        expression_level_analysis3(adata.copy(), outdir=f"{outdir}/gex3", k=k)

    for k in [8, 16, 32]:
        adata = antigen_analysis(adata, outdir=f"{outdir}/protein_hist", k=k)
    for k in [8]:
        visualize_tcr_umap(adata, outdir=f"{outdir}/tcr_cluster_boxplot_k{k}", k=k, feature='blosum', n_proc=30,
                           legend=True, consider_corona=True, sample_data=True)

    return adata


def PA_ratio_analysis(adata, outdir="analysis/240913_PA_ratio_analysis", swap_axes=False):
    # If swap_axes is True, the plot considers cell type on a larger scale
    df = adata.obs
    df = clean_wos_get_single_pair(df)

    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    def calculate_PA_ratio_with_stats(data, reference_data, match_columns=None):
        """
        Calculate PA ratio and perform statistical comparison with reference data.

        Args:
            data: DataFrame for current pattern
            reference_data: DataFrame for reference pattern (naive/healthy)
            match_columns: List of match columns to consider

        Returns:
            ratio: PA ratio
            p_value: p-value from Fisher's exact test
            PA_count: Number of PA cells
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

    def create_heatmap(data, title, output_path):
        plt.figure(figsize=(12, 8))
        sns.heatmap(data, annot=True, fmt='.2f', cmap='YlOrRd', vmin=0, vmax=1, cbar_kws={'label': 'PA Ratio'})
        plt.title(title)
        plt.xlabel('Cell Types')
        plt.ylabel('Site Patterns')
        plt.tight_layout()
        plt.savefig(output_path, format='pdf', dpi=300)
        plt.close()

    def create_pa_ratio_bar_plot(data, p_value_df, title, output_path, alpha=0.05, swap_axes=False):
        """
        Create a bar plot of PA ratios with significance markers comparing to healthy/naive.
        Args:
            data: DataFrame with PA ratios
            p_value_df: DataFrame with corrected p-values
            title: plot title
            output_path: path to save the plot
            alpha: significance threshold
            swap_axes: if True, cell types become main groups and WHO scales become subgroups
        """
        # Melt the dataframe to long format
        melted_data = data.reset_index().melt(id_vars='index',
                                             var_name='cell_type',
                                             value_name='PA Ratio')
        melted_data = melted_data.rename(columns={'index': 'Who Ordinal Scale'})

        # Get unique cell types and patterns
        patterns = melted_data['Who Ordinal Scale'].unique()

        # Set up the plot
        fig, ax = plt.subplots(figsize=(14, 7))
        bar_width = 0.15
        gap_width = 0.02

        # Determine primary and secondary grouping based on swap_axes
        if swap_axes:
            primary_groups = CELL_NAMES
            secondary_groups = patterns
            primary_label = 'Cell Type'
            secondary_label = 'Who Ordinal Scale'
        else:
            primary_groups = patterns
            secondary_groups = CELL_NAMES
            primary_label = 'Who Ordinal Scale'
            secondary_label = 'Cell Type'

        group_width = len(secondary_groups) * (bar_width + gap_width) - gap_width

        # Calculate positions for each group of bars
        group_positions = np.arange(len(primary_groups))
        bar_positions_dict = {}

        # Plot bars
        max_ratio = melted_data['PA Ratio'].max()

        for i, secondary_group in enumerate(secondary_groups):
            if swap_axes:
                group_data = melted_data[melted_data[secondary_label] == secondary_group]
                bar_positions = group_positions + i * (bar_width + gap_width) - group_width / 2 + bar_width / 2
                bars = ax.bar(bar_positions, group_data['PA Ratio'],
                              width=bar_width, label=secondary_group,
                              edgecolor='black', linewidth=1)
                bar_positions_dict[secondary_group] = bar_positions

                # Add significance markers
                for j, cell_type in enumerate(primary_groups):
                    if secondary_group != 'healthy':  # Skip reference condition
                        p_value = p_value_df[cell_type][secondary_group]
                        if not np.isnan(p_value) and p_value < alpha:
                            current_height = group_data[group_data['cell_type'] == cell_type]['PA Ratio'].iloc[0]
                            # Add significance markers
                            if p_value < 0.001:
                                marker = '***'
                            elif p_value < 0.01:
                                marker = '**'
                            elif p_value < 0.05:
                                marker = '*'
                            # Position the marker above the bar
                            y_pos = current_height + (max_ratio * 0.02)
                            ax.text(bar_positions[j], y_pos, marker,
                                   ha='center', va='bottom', fontsize=12)
            else:
                group_data = melted_data[melted_data['cell_type'] == secondary_group]
                bar_positions = group_positions + i * (bar_width + gap_width) - group_width / 2 + bar_width / 2
                bars = ax.bar(bar_positions, group_data['PA Ratio'],
                             width=bar_width, label=secondary_group,
                             edgecolor='black', linewidth=1)
                bar_positions_dict[secondary_group] = bar_positions

                # Add significance markers
                for j, pattern in enumerate(primary_groups):
                    if pattern != 'healthy':  # Skip reference condition
                        p_value = p_value_df.loc[pattern, secondary_group]
                        if not np.isnan(p_value) and p_value < alpha:
                            current_height = group_data[group_data['Who Ordinal Scale'] == pattern]['PA Ratio'].values[0]

                            # Add significance markers
                            if p_value < 0.001:
                                marker = '***'
                            elif p_value < 0.01:
                                marker = '**'
                            elif p_value < 0.05:
                                marker = '*'
                            # Position the marker above the bar
                            y_pos = current_height + (max_ratio * 0.02)
                            ax.text(bar_positions[j], y_pos, marker,
                                   ha='center', va='bottom', fontsize=12)

        # Customize the plot
        ax.set_xlabel(primary_label, fontsize=14)
        ax.set_ylabel('PA Ratio', fontsize=14)
        ax.set_xticks(group_positions)
        ax.set_xticklabels(primary_groups, rotation=0, ha='center', fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_ylim(0, max_ratio * 1.15)

        # Add legend
        ax.legend(title=secondary_label,
                 bbox_to_anchor=(1.05, 1),
                 loc='upper left',
                 fontsize=14,
                 title_fontsize=14)

        # Add significance level explanation
        ax.text(1.05, 0.5,
                '* p < 0.05\n** p < 0.01\n*** p < 0.001\nCompared to healthy',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

    # Experiment with match columns (may remove this for loop later)
    for k in [8, 16, 32]:
        print(f"Processing with k={k}")
        match_cols = [f'match_{i}' for i in range(k)]

        PA_ratio_data = []
        PA_num_data = []
        num_data = []
        p_values_data = []

        for i, pattern in enumerate(WOS_PATTERNS):
            row_ratio = []
            row_PA_num = []
            row_num = []
            row_pvals = []

            for j, idents in enumerate(CELL_TYPES):
                df_filtered = df[(df['Who Ordinal Scale'].isin(pattern)) &
                               (df['leiden'].isin(idents))]

                # Get reference data (healthy/naive) for this cell type
                df_ref = df[(df['Who Ordinal Scale'].isin(WOS_PATTERNS[0])) &
                           (df['leiden'].isin(idents))]

                ratio, p_value, PA_num = calculate_PA_ratio_with_stats(
                    df_filtered, df_ref, match_cols)

                row_ratio.append(ratio)
                row_PA_num.append(PA_num)
                row_num.append(len(df_filtered))
                row_pvals.append(p_value)

            PA_ratio_data.append(row_ratio)
            PA_num_data.append(row_PA_num)
            num_data.append(row_num)
            p_values_data.append(row_pvals)

        # Create DataFrames
        PA_ratio_df = pd.DataFrame(
            PA_ratio_data,
            columns=CELL_NAMES,
            index=PATTERN_NAMES
        )

        p_value_df = pd.DataFrame(
            p_values_data,
            columns=CELL_NAMES,
            index=PATTERN_NAMES
        )

        PA_num_df = pd.DataFrame(
            PA_num_data,
            columns=CELL_NAMES,
            index=PATTERN_NAMES
        )

        num_df = pd.DataFrame(
            num_data,
            columns=CELL_NAMES,
            index=PATTERN_NAMES
        )

        # Apply multiple testing correction within each cell type
        corrected_pvals = []
        for col in range(p_value_df.shape[1]):
            col_pvals = p_value_df.iloc[1:, col]  # Skip healthy/naive row
            mask = ~np.isnan(col_pvals)
            if mask.sum() > 0:
                _, corrected, _, _ = multipletests(col_pvals[mask], method='fdr_bh')
                corrected_col = np.full(len(col_pvals), np.nan)
                corrected_col[mask] = corrected
                # Add back the healthy/naive row with NaN
                corrected_col = np.insert(corrected_col, 0, np.nan)
                corrected_pvals.append(corrected_col)
            else:
                corrected_pvals.append(np.full(p_value_df.shape[0], np.nan))

        corrected_p_value_df = pd.DataFrame(
            np.array(corrected_pvals).T,
            columns=CELL_NAMES,
            index=PATTERN_NAMES
        )

        if swap_axes:
            output_path = f'{outdir}/PA_ratio_swap_{k}.pdf'
        else:
            output_path = f'{outdir}/PA_ratio_{k}.pdf'
        create_pa_ratio_bar_plot(
            PA_ratio_df,
            corrected_p_value_df,
            'PA Ratio - leiden and Who Ordinal Scale',
            output_path,
            swap_axes=swap_axes
        )

        # Save the core data
        PA_ratio_df.to_csv(f"{outdir}/PA_ratio_df_k{k}.csv", index=False)
        p_value_df.to_csv(f"{outdir}/p_value_df_k{k}.csv", index=False)
        PA_num_df.to_csv(f"{outdir}/PA_num_df_k{k}.csv", index=False)
        num_df.to_csv(f"{outdir}/num_df_k{k}.csv", index=False)

        print(f"PA ratio analysis completed. Results saved in {outdir}")
    return adata


def perform_deg_analysis(adata_subset, genes, match_columns):
    """
    Perform DEG analysis between PA and NA cells using scanpy's rank_genes_groups.

    Parameters:
    adata_subset (AnnData): Subset of the annotated data matrix.
    genes (list): List of genes of interest.

    Returns:
    dict: DEG results for the genes of interest.
    """
    # Create PA/NA grouping
    adata_subset.obs['comparison_group'] = 'NA'  # Default to NA
    pa_mask = adata_subset.obs[match_columns].apply(
        lambda row: any(row[col] == 1 for col in match_columns), axis=1
    )
    adata_subset.obs.loc[pa_mask, 'comparison_group'] = 'PA'

    # Perform DEG analysis
    try:
        sc.tl.rank_genes_groups(
            adata_subset,
            groupby='comparison_group',
            groups=['PA'],  # Compare PA vs rest (NA)
            reference='NA',
            method='wilcoxon',
            key_added='deg_results',
            pts=True  # Calculate percentage of cells expressing genes
        )
    except:
        print("No PA T cells in this group were found, so skip DEG analysis.")
        return None

    # Extract results
    results = {gene: {} for gene in genes}
    for gene in genes:
        if gene in adata_subset.var_names:
            try:
                idx = [x[0] for x in adata_subset.uns['deg_results']['names']].index(gene)
                results[gene] = {
                    'logfoldchange': [x[0] for x in adata_subset.uns['deg_results']['logfoldchanges']][idx],
                    'pval': [x[0] for x in adata_subset.uns['deg_results']['pvals']][idx],
                    'pval_adj': [x[0] for x in adata_subset.uns['deg_results']['pvals_adj']][idx]
                }
            except:
                print(f"Cannot find DEG result for the gene: {gene}. Exit program..")
                breakpoint()
                exit(0)

    return results


def expression_level_analysis(adata, outdir, k=0):
    """
    Perform differential gene expression analysis between PA and NA cells for each WOS pattern
    and generate heatmaps of expression differences.

    Parameters:
    adata (AnnData): Annotated data matrix.
    outdir (str): Directory to save the output files.
    k (int): Parameter for generating match columns (default: 0).
    """
    def create_grouped_gene_expression_plot(genes_of_interest, gene_groups, deg_results, pattern_name, outdir, k):
        """
        Create a grouped bar plot of gene expression differences with significance markers.
        Genes are grouped by their functional categories. Bars are colored red for positive differences and blue for negative differences.
        Y-axis is fixed between -0.7 and 0.7.

        Parameters:
        genes_of_interest (list): List of genes to include in the plot.
        gene_groups (dict): Dictionary mapping gene groups to lists of genes.
        deg_results (dict): DEG results from `perform_deg_analysis`.
        pattern_name (str): Name of the WOS pattern being analyzed.
        outdir (str): Directory to save the output plot.
        k (int): Parameter for generating match columns.
        """
        # Extract gene data from DEG results
        gene_data = {}
        for gene in genes_of_interest:
            if gene in deg_results:
                gene_data[gene] = {
                    'diff': deg_results[gene]['logfoldchange'],
                    'pval_adj': deg_results[gene]['pval_adj']  # Use adjusted p-values from scanpy
                }

        # Create plot
        plt.figure(figsize=(15, 10))
        plt.ylim(-0.7, 0.7)

        # Calculate positions for grouped bars
        group_positions = []
        current_pos = 0

        for group_name in gene_groups.keys():
            group_genes = gene_groups[group_name]
            group_size = len([g for g in group_genes if g in gene_data])
            if group_size > 0:
                group_positions.append((group_name, current_pos, current_pos + group_size))
                current_pos += group_size + 1.5  # Add space between groups

        bar_positions = []
        bar_labels = []

        # Fixed y_offset based on the ylim range
        y_offset = 0.05  # This is about 7% of the total range (-0.7 to 0.7 = 1.4)

        for group_name, start_pos, end_pos in group_positions:
            group_genes = [g for g in gene_groups[group_name] if g in gene_data]
            positions = np.arange(start_pos, start_pos + len(group_genes))
            diffs = [gene_data[g]['diff'] for g in group_genes]

            # Create bars with colors based on difference direction
            for pos, diff, gene in zip(positions, diffs, group_genes):
                color = 'red' if diff >= 0 else 'blue'
                plt.bar(pos, diff, color=color, alpha=0.7)

            # Add significance markers
            for pos, gene in zip(positions, group_genes):
                height = gene_data[gene]['diff']
                stars = get_stars(gene_data[gene]['pval_adj'])  # Use adjusted p-values

                # Adjust text position based on bar direction
                if height >= 0:
                    text_y = min(height + y_offset, 0.68)  # Cap at 0.65 to stay within ylim
                else:
                    text_y = max(height - y_offset, -0.68)  # Cap at -0.65 to stay within ylim

                plt.text(pos, text_y, stars,
                         ha='center', va='bottom' if height >= 0 else 'top')

            bar_positions.extend(positions)
            bar_labels.extend(group_genes)

            # Add group labels
            group_center = np.mean(positions)
            plt.text(group_center, -0.85,  # Position labels below the plot
                     group_name, ha='center', va='top', rotation=90)

        # Customize plot
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        plt.title(f"Differential Gene Expression by Functional Group (PA - NA)\n{pattern_name}",
                  fontsize=12, pad=20)
        plt.xlabel("Genes", fontsize=10)
        plt.ylabel("Expression Difference", fontsize=10)

        # Set x-axis ticks and labels
        plt.xticks(bar_positions, bar_labels, rotation=90, ha='center')

        # Add legend for significance levels
        significance_elements = [
            plt.Text(0, 0, '*** p ≤ 0.001'),
            plt.Text(0, 0, '** p ≤ 0.01'),
            plt.Text(0, 0, '* p ≤ 0.05'),
            plt.Text(0, 0, 'ns p > 0.05')
        ]

        # Add legend for bar colors
        color_elements = [
            plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.7, label='Upregulated in PA'),
            plt.Rectangle((0, 0), 1, 1, fc='blue', alpha=0.7, label='Downregulated in PA')
        ]

        # Create legends
        plt.legend(handles=color_elements, bbox_to_anchor=(1.05, 1),
                   loc='upper left', title='Expression Change')
        sig_legend = plt.legend(handles=significance_elements, bbox_to_anchor=(1.05, 0.6),
                                loc='center left', title='Adjusted Significance Levels')
        plt.gca().add_artist(sig_legend)

        # Adjust layout to accommodate the fixed y-axis range and group labels
        plt.subplots_adjust(bottom=0.2)  # Make room for group labels

        # Save plot
        plt.savefig(os.path.join(outdir, f"grouped_gene_exp_diff_{pattern_name}_k{k}.pdf"),
                    format='pdf', bbox_inches='tight', dpi=300)
        plt.close()

        return gene_data

    def plot_absolute_gene_expressions(genes, cell_names, mean_expressions, p_values, pattern_name, outdir, k,
                                       key_genes=['PRF1', 'GZMB']):
        """
        Create visualizations comparing PA and NA gene expressions for each cell type.

        Parameters:
        - genes: List of genes in desired order
        - cell_names: List of cell types
        - mean_expressions: Dict containing PA and NA expression values
        - p_values: Dict containing p-values for each comparison
        - pattern_name: Name of the pattern for plot titles
        - outdir: Output directory
        - k: Parameter k for filename
        - key_genes: List of genes to highlight in the plots
        """
        for cell_name in cell_names:
            plt.figure(figsize=(12, 6))

            # Prepare data for plotting
            pa_values = []
            na_values = []
            significant = []

            for gene in genes:
                if gene in mean_expressions['PA'] and gene in mean_expressions['NA']:
                    pa_values.append(mean_expressions['PA'][gene])
                    na_values.append(mean_expressions['NA'][gene])
                    significant.append(p_values[gene]['pval_adj'] < 0.05)  # Use adjusted p-values
                else:
                    pa_values.append(0)
                    na_values.append(0)
                    significant.append(False)

            # Create paired bar plot
            x = np.arange(len(genes))
            width = 0.35

            fig, ax = plt.subplots(figsize=(15, 6))
            rects1 = ax.bar(x - width/2, pa_values, width, label='PA', color='#ff7f0e', alpha=0.7)
            rects2 = ax.bar(x + width/2, na_values, width, label='NA', color='#1f77b4', alpha=0.7)

            # Add significance markers
            for i, (pa, na, sig) in enumerate(zip(pa_values, na_values, significant)):
                if sig:
                    max_val = max(pa, na)
                    ax.text(i, max_val + 0.1, '*', ha='center', va='bottom', color='red')

            # Customize plot
            ax.set_ylabel('Expression Level')
            ax.set_title(f'Gene Expression Comparison in {cell_name}\n{pattern_name}')
            ax.set_xticks(x)
            ax.set_xticklabels(genes, rotation=45, ha='right')
            ax.legend()

            # Highlight key genes
            for i, gene in enumerate(genes):
                if gene in key_genes:
                    ax.axvspan(i-0.5, i+0.5, alpha=0.2, color='yellow')

            # Add grid
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)

            # Adjust layout
            plt.tight_layout()

            # Save plot
            plt.savefig(os.path.join(outdir, f"gene_exp_{cell_name}_{pattern_name}_k{k}.pdf"))
            plt.close()

    def visualize_heatmap(diff_matrix, p_value_matrix, genes, cell_names, pattern_name, outdir, k, pattern_samples, pa_samples):
        """
        Visualize a heatmap of gene expression differences between PA and NA cells, per cell type.

        Parameters:
        - diff_matrix (list of lists): Matrix of expression differences (PA - NA). Rows: genes, Columns: cell types.
        - p_value_matrix (list of lists): Matrix of adjusted p-values. Rows: genes, Columns: cell types.
        - genes (list): List of genes to include in the heatmap.
        - cell_names (list): List of cell types to include in the heatmap.
        - pattern_name (str): Name of the WOS pattern for the plot title.
        - outdir (str): Directory to save the output plot.
        - k (int): Parameter for generating the filename.
        - pattern_samples (int): Total number of samples for the current WOS pattern.
        - pa_samples (int): Number of PA samples for the current WOS pattern.
        """
        # Format x-axis labels to include sample information
        x_labels = [f"{cell_name}\n(N={pattern_samples}, PA={pa_samples[cell_name]})" for cell_name in cell_names]

        # Create summary heatmap
        plt.figure(figsize=(12, 10))
        heatmap = sns.heatmap(diff_matrix, cmap="RdBu_r", center=0,
                              xticklabels=x_labels, yticklabels=genes,
                              annot=True, fmt=".2f", cbar_kws={'label': 'Expression Difference (PA - NA)'})

        # Highlight cells with p-value < 0.05
        for i in range(len(genes)):
            for j in range(len(cell_names)):
                try:
                    if p_value_matrix[i][j] < 0.05:
                        heatmap.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
                except:
                    breakpoint()

        plt.title(f"Gene Exp Diff (PA - NA) [{pattern_name}]")
        plt.xlabel("Cell Types")
        plt.ylabel("Genes")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"gene_diff_heatmap_{pattern_name}_k{k}.pdf"), format='pdf')
        plt.close()

    print("Stringify Who Ordinal Scale..")
    adata.obs['Who Ordinal Scale'] = adata.obs['Who Ordinal Scale'].apply(clean_wos_value)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print("Focus on chain_pairing == Single pair..")
    adata = adata[adata.obs['chain_pairing'] == 'Single pair']
    print(f"(expression_level_analysis()): Total number of rows: {len(adata.obs)}")

    os.makedirs(outdir, exist_ok=True)

    # Main axes of analysis
    wos_patterns = [ALL] + WOS_PATTERNS
    pattern_names = ['all'] + PATTERN_NAMES
    cell_names = CELL_NAMES

    # Generate column names based on k
    match_columns = [f'match_{i}' for i in range(k)]

    for wos_pattern, pattern_name in zip(wos_patterns, pattern_names):
        print(f"Processing WOS pattern: {pattern_name}")

        # Filter data for the current WOS pattern
        adata_subset = adata[adata.obs['Who Ordinal Scale'].isin(wos_pattern)]

        # Create PA/NA grouping
        adata_subset.obs['comparison_group'] = 'NA'  # Default to NA
        pa_mask = adata_subset.obs[match_columns].apply(
            lambda row: any(row[col] == 1 for col in match_columns), axis=1
        )
        adata_subset.obs.loc[pa_mask, 'comparison_group'] = 'PA'

        # Get total number of samples for this pattern
        pattern_samples = len(adata_subset)

        # Perform aggregate DEG analysis between PA and NA cells
        aggregate_deg_results = perform_deg_analysis(adata_subset, GENES_OF_INTEREST, match_columns)

        if aggregate_deg_results is None:
            print(f"No PA T cells found for WOS pattern {pattern_name}. Skipping DEG analysis.")
            continue

        # Prepare data for heatmap (per cell type)
        diff_matrix = []  # Rows: genes, Columns: cell types
        p_value_matrix = []  # Rows: genes, Columns: cell types
        pa_count = {}

        for celltype, cell_name in zip(CELL_TYPES, CELL_NAMES):
            # Filter data for the current cell type
            adata_celltype = adata[(adata.obs['leiden'].isin(celltype)) & (adata.obs['Who Ordinal Scale'].isin(wos_pattern))]
            # Create PA / NA grouping
            adata_celltype.obs['comparison_group'] = 'NA'
            pa_mask = adata_celltype.obs[match_columns].apply(
                lambda row: any(row[col] == 1 for col in match_columns), axis=1)
            adata_celltype.obs.loc[pa_mask, 'comparison_group'] = 'PA'

            # Get number of PA samples per cell name
            pa_count[cell_name] = len(adata_celltype[(adata_celltype.obs['comparison_group'] == 'PA')])

            # Perform DEG analysis for the current cell type
            print(f"Perform DEG analysis for cell_name: {cell_name}..")
            celltype_deg_results = perform_deg_analysis(adata_celltype.copy(), GENES_OF_INTEREST, match_columns)

            gene_diffs = []
            gene_pvals = []
            for gene in GENES_OF_INTEREST:
                if celltype_deg_results and gene in celltype_deg_results:
                    # Extract log fold change and adjusted p-value for the gene
                    gene_diffs.append(celltype_deg_results[gene]['logfoldchange'])
                    gene_pvals.append(celltype_deg_results[gene]['pval_adj'])
                else:
                    # If the gene is not in DEG results, use default values
                    gene_diffs.append(0.0)
                    gene_pvals.append(1.0)
            diff_matrix.append(gene_diffs)
            p_value_matrix.append(gene_pvals)

        diff_matrix = np.array(diff_matrix).T
        p_value_matrix = np.array(p_value_matrix).T

        # Visualize heatmap (x-axis: cell types)
        visualize_heatmap(
            diff_matrix=diff_matrix,
            p_value_matrix=p_value_matrix,
            genes=GENES_OF_INTEREST,
            cell_names=CELL_NAMES,
            pattern_name=pattern_name,
            outdir=outdir,
            k=k,
            pattern_samples=pattern_samples,
            pa_samples=pa_count
        )

        # Create grouped gene expression plot using aggregate DEG results
        create_grouped_gene_expression_plot(
            genes_of_interest=GENES_OF_INTEREST,
            gene_groups=GENE_GROUPS,
            deg_results=aggregate_deg_results,
            pattern_name=pattern_name,
            outdir=outdir,
            k=k
        )

        # # Collect mean expressions for PA and NA cells
        # mean_expressions = {'PA': {}, 'NA': {}}
        # for gene in GENES_OF_INTEREST:
        #     if gene in aggregate_deg_results:
        #         mean_expressions['PA'][gene] = 0
        #         mean_expressions['NA'][gene] = 0  # NA is the reference group, so its logfoldchange is 0

        # # Plot absolute gene expressions
        # plot_absolute_gene_expressions(
        #     genes=GENES_OF_INTEREST,
        #     cell_names=CELL_NAMES,
        #     mean_expressions=mean_expressions,
        #     p_values=aggregate_deg_results,
        #     key_genes=['GZMB', 'PRF1'],
        #     pattern_name=pattern_name,
        #     outdir=outdir,
        #     k=k
        # )

    return adata


def antigen_analysis(adata, outdir, k=1, consider_celltype=False):
    """
    Track the protein sources of generated epitopes per patient group

    Parameters
    ----------
    adata: AnnData
        scanpy object
    outdir: str
        output directory
    k: int
        how many top generations to consider for matching (i.e. match_{k-1})
    consider_celltype: bool
        if True, the analysis is done by per (patient group, celltype)
        but this case, some cases may lack data
    """
    df = adata.obs
    df = clean_wos_get_single_pair(df)

    df['celltype'] = df['leiden'].map(LEIDEN2CELLNAME)

    who2pattern = {
        'nan': 'healthy',
        '1': 'mild', '1 or 2': 'mild', '2': 'mild',
        '3': 'moderate', '4': 'moderate',
        '5': 'severe', '6': 'severe', '7': 'severe'
    }
    df['pattern'] = df['Who Ordinal Scale'].map(who2pattern)

    Path(outdir).mkdir(parents=True, exist_ok=True)

    def trim_prot(protein_name):
        if isinstance(protein_name, str):
            if '[Severe acute respiratory syndrome coronavirus 2]' in protein_name:
                protein_name = protein_name.split("[")[0].strip()

            # Standardize name
            protein_name = str(protein_name).lower()  # Convert to lowercase for consistent matching

            if any(x in protein_name for x in ['orf1ab', 'replicase']):
                return 'Non-structural proteins (NSP)'
            elif any(x in protein_name for x in ['orf']):
                return 'Accessory proteins (ORFs)'
            elif any(x in protein_name for x in ['surface', 'spike', 'glycoprotein']):
                return 'Spike (S) protein'
            elif any(x in protein_name for x in ['nucleocapsid', 'nucleoprotein']):
                return 'Nucleocapsid (N) protein'
            elif any(x in protein_name for x in ['membrane']):
                return 'Membrane (M) protein'
            elif any(x in protein_name for x in ['envelope', 'envelop']):
                return 'Envelope (E) protein'
            else:
                return 'Other'
        return protein_name

    # Clean up the protein names and keep only the first match for each row
    conditions = [df[f'match_{i}'] == 1 for i in range(k)]
    df = df[pd.concat(conditions, axis=1).any(axis=1)]

    # Create a new column to store the first matching protein for each row
    df['first_matching_protein'] = None
    for i in range(k):
        # Only update rows where first_matching_protein is still None and match_i is 1
        mask = (df['first_matching_protein'].isna()) & (df[f'match_{i}'] == 1)
        df.loc[mask, 'first_matching_protein'] = df.loc[mask, f'ref_protein_{i}'].apply(lambda x: trim_prot(x))

    # Create the long-form DataFrame using only the first matching protein
    df_long = df[['clonotype_size', 'pattern', 'celltype', 'first_matching_protein']].copy()
    df_long = df_long.rename(columns={'first_matching_protein': 'ref_protein'})

    # Remove any rows where ref_protein is missing or NaN
    df_long = df_long.dropna(subset=['ref_protein'])

    def normalize_data(data: pd.DataFrame,
                      consider_celltype: bool = True) -> pd.DataFrame:
        """
        Normalize the protein distribution data using various methods.

        Args:
            data: Long-form DataFrame with columns: ref_protein, pattern, celltype, clonotype_size
            consider_celltype: Whether to consider cell types in normalization
        """
        # Group by relevant columns to get counts
        group_cols = ['ref_protein', 'pattern', 'celltype'] if consider_celltype else ['ref_protein', 'pattern']
        counts = data.groupby(group_cols)['clonotype_size'].count().reset_index()

        # Calculate percentages within each pattern (and celltype if considered)
        norm_cols = ['pattern', 'celltype'] if consider_celltype else ['pattern']
        totals = counts.groupby(norm_cols)['clonotype_size'].transform('sum')
        counts['normalized_value'] = (counts['clonotype_size'] / totals) * 100
        return counts

    def plot_protein_distribution(data: pd.DataFrame,
                                title: str,
                                protein2color: dict,
                                filename: str,
                                consider_celltype: bool = True,
                                top_n: int = 20,
                                outdir: str = '.') -> None:
        """
        Create stacked bar plots showing protein distribution across severity levels.

        Parameters:
        - data: DataFrame with columns for proteins, patterns (severity), celltypes, and frequencies
        - title: Plot title
        - filename: Output filename
        - consider_celltype: Whether to split by cell type
        - top_n: Number of top proteins to include
        - outdir: Output directory
        """
        # Normalize the data
        norm_data = normalize_data(data, consider_celltype=consider_celltype)

        pattern_counts = data.groupby('pattern').size()

        # Get top N proteins by total frequency if specified
        if top_n is not None:
            top_proteins = (norm_data.groupby('ref_protein')['clonotype_size']
                           .sum()
                           .sort_values(ascending=False)
                           .head(top_n)
                           .index)
            norm_data = norm_data[norm_data['ref_protein'].isin(top_proteins)]

        # Create pivot table
        if consider_celltype:
            pivot = pd.pivot_table(norm_data,
                                 values='normalized_value',
                                 index=['pattern', 'celltype'],
                                 columns='ref_protein')
        else:
            pivot = pd.pivot_table(norm_data,
                                 values='normalized_value',
                                 index='pattern',
                                 columns='ref_protein')
        pivot = pivot.fillna(0)

        # Sort severity levels
        severity_order = ['healthy', 'mild', 'moderate', 'severe']
        pivot = pivot.reindex(severity_order)

        # Create the stacked bar plot
        plt.figure(figsize=(12, 6))
        ax = plt.gca()

        # Plot bars
        bottom = np.zeros(len(pivot))
        for protein in pivot.columns:
            values = pivot[protein].values
            ax.bar(pivot.index, values, bottom=bottom,
                   label=protein, color=protein2color[protein])
            bottom += values

        # Customize plot
        plt.title(title, pad=20)
        x_labels = []
        for pattern in pivot.index:
            try:
                x_labels.append(f'{pattern}\n(N={pattern_counts[pattern]})')
            except:
                x_labels.append(f'{pattern}\n(N=0)')
        plt.xticks(range(len(pivot.index)), x_labels)

        plt.xlabel('Disease Severity')
        plt.ylabel('Percentage (%)')

        # Add legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                  borderaxespad=0., title='Target Proteins')

        # Add percentage labels on bars
        for idx, severity in enumerate(pivot.index):
            total = 0
            for protein in pivot.columns:
                value = pivot.loc[severity, protein]
                if value > 0:  # Only add label if value is significant
                    plt.text(idx, total + value/2, f'{value:.1f}%',
                            ha='center', va='center')
                total += value

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{outdir}/{filename}.pdf", bbox_inches='tight', dpi=300)
        plt.close()

    # Plot heatmap for clonally expanded (CE) T cells
    plot_protein_distribution(df_long[df_long['clonotype_size'] > 1],
                            "Protein Distribution (% within severity)",
                              PROTEIN2COLOR,
                            f"{outdir}/protein_ratio_CE_k{k}",
                            consider_celltype=consider_celltype)

    # Plot heatmap for not clonally expanded (NE) T cells
    plot_protein_distribution(df_long[df_long['clonotype_size'] == 1],
                            "Protein Distribution (% within severity)",
                              PROTEIN2COLOR,
                            f"{outdir}/protein_ratio_NE_k{k}",
                            consider_celltype=consider_celltype)

    print("Analysis complete. Figures saved in the output directory.")
    return adata


def blosum62_score(pair):
    a, b = pair
    return blosum62.get((a, b), blosum62.get((b, a), -4))

def sequence_distance(seq_pair):
    seq1, seq2 = seq_pair
    return sum(blosum62_score((a, b)) for a, b in zip(seq1, seq2))

def pairwise_blosum62_distance(peptides, n_proc=1):
    global blosum62
    blosum62 = substitution_matrices.load('BLOSUM62')

    n_peptides = len(peptides)
    distances = np.zeros((n_peptides, n_peptides))

    if n_proc > 1:
        with mp.Pool(processes=n_proc) as pool:
            results = pool.map(sequence_distance, combinations(peptides, 2))

        idx = np.triu_indices(n_peptides, k=1)
        distances[idx] = results
        distances = distances + distances.T
    else:
        for i, j in combinations(range(n_peptides), 2):
            distance = sequence_distance((peptides[i], peptides[j]))
            distances[i, j] = distance
            distances[j, i] = distance

    return distances


def visualize_tcr_umap(adata, outdir, k=2, feature='blosum', n_proc=1, legend=True, consider_corona=True, sample_data=True):
    """
    Visualize TCRs in UMAP space with clone sizes and matched status highlighting.

    Parameters:
    - adata: AnnData object containing TCR data
    - outdir: Output directory for saving plots
    - k: Number of match columns to consider
    - feature: Feature type for distance calculation ('blosum' by default)
    - n_proc: Number of processes for parallel computation
    - legend: Whether to show legend
    """
    def calculate_clone_sizes(data):
        return data.groupby('clonotype').size()

    # Create output directory
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Preprocess data
    df = adata.obs
    df = clean_wos_get_single_pair(df)

    # Calculate clone sizes
    clone_sizes = calculate_clone_sizes(df)
    df['clone_size'] = df['clonotype'].map(clone_sizes)

    # Define severity patterns
    def analyze_clone_sizes(df, wos_patterns, pattern_names, k, outdir):
        """
        Analyze and visualize clone sizes for different TCR groups across disease patterns.
        Performs statistical testing against the healthy group with multiple testing correction.

        Args:
            df: DataFrame containing TCR data
            wos_patterns: List of WHO Ordinal Scale patterns
            pattern_names: List of pattern names corresponding to WOS patterns
            k: Number of patterns to consider for matching
            outdir: Output directory for saving results
        """
        # Initialize lists to store results
        results = []

        # Create directory if it doesn't exist
        Path(outdir).mkdir(parents=True, exist_ok=True)

        # Process each pattern
        for wos_pattern, pattern_name in zip(wos_patterns, pattern_names):
            pattern_df = df[df['Who Ordinal Scale'].isin(wos_pattern)].copy()
            stat_df = pattern_df.copy()

            # Determine COVID-19 matched status
            match_columns = [f'match_{i}' for i in range(k)]
            stat_df['matched'] = stat_df[match_columns].apply(
                lambda row: any(row[col] == 1 for col in match_columns), axis=1)

            # Determine coronavirus status
            corona_columns = [f'corona_{i}' for i in range(k)]
            stat_df['corona'] = stat_df[corona_columns].apply(
                lambda row: any(row[col] == 1 for col in corona_columns), axis=1)

            # Get unique clonotypes and their clone sizes
            background_sizes = (stat_df[~stat_df['matched']]
                              .groupby('clonotype')['clone_size']
                              .first()  # Take the first occurrence of clone size
                              .values)

            covid_sizes = (stat_df[stat_df['matched']]
                          .groupby('clonotype')['clone_size']
                          .first()
                          .values)

            corona_sizes = (stat_df[stat_df['corona']]
                           .groupby('clonotype')['clone_size']
                           .first()
                           .values)

            # Calculate statistics
            result = {
                'pattern': pattern_name,
                'background_sizes': background_sizes,
                'covid_sizes': covid_sizes,
                'corona_sizes': corona_sizes
            }
            results.append(result)

        def create_comparison_plot(tcr_type, color, save_plot=True):
            fig, ax1 = plt.subplots(figsize=(8, 6))

            # Create right axis for original scale means
            ax2 = ax1.twinx()

            # Initialize lists to store statistics
            stats_data = []

            # Reference (healthy) group
            healthy_sizes = results[0][f'{tcr_type}_sizes']

            # Calculate statistics for each group
            for idx, result in enumerate(results):
                current_sizes = result[f'{tcr_type}_sizes']
                log_sizes = [math.log(x, 10) for x in current_sizes]

                # Calculate basic statistics
                if len(current_sizes) == 0:  # ex) happens when there is zero COVID19 samples in 'healthy' pattern
                    stats_dict = {
                        'group': result['pattern'],
                        'tcr_type': tcr_type,
                        'n_clones': len(current_sizes),
                        'mean_raw': 0,
                        'median_raw': 0,
                        'mean_log10': 0,
                        'median_log10': 0,
                        'q1_log10': 0,
                        'q3_log10': 0,
                        'iqr_log10': 0,
                        'whisker_low_log10': 0,
                        'whisker_high_log10': 0,
                        'log_sizes': [],
                        'pvalue_vs_healthy': 1.0,
                        'pvalue_adjusted': 1.0
                    }
                    stats_data.append(stats_dict)
                else:
                    stats_dict = {
                        'group': result['pattern'],
                        'tcr_type': tcr_type,
                        'n_clones': len(current_sizes),
                        'mean_raw': np.mean(current_sizes),
                        'median_raw': np.median(current_sizes),
                        'mean_log10': np.mean(log_sizes),
                        'median_log10': np.median(log_sizes),
                        'q1_log10': np.percentile(log_sizes, 25),
                        'q3_log10': np.percentile(log_sizes, 75),
                        'iqr_log10': np.percentile(log_sizes, 75) - np.percentile(log_sizes, 25),
                        'whisker_low_log10': np.percentile(log_sizes, 25) - (1.5 * (np.percentile(log_sizes, 75) - np.percentile(log_sizes, 25))),
                        'whisker_high_log10': np.percentile(log_sizes, 75) + (1.5 * (np.percentile(log_sizes, 75) - np.percentile(log_sizes, 25)))
                    }
                    stats_dict['log_sizes'] = ';'.join(map(str, log_sizes))  # Store as semicolon-separated string

                    # Calculate p-values vs healthy
                    if result['pattern'] == 'healthy':
                        stats_dict['pvalue_vs_healthy'] = 1.0
                        stats_dict['pvalue_adjusted'] = 1.0
                    else:
                        if len(healthy_sizes) == 0:
                            p_val = 1.0
                        else:
                            stat, p_val = mannwhitneyu(current_sizes, healthy_sizes, alternative='greater')
                        stats_dict['pvalue_vs_healthy'] = p_val

                    stats_data.append(stats_dict)

            # Create DataFrame
            stats_df = pd.DataFrame(stats_data)

            # Perform multiple testing correction
            mask = stats_df['group'] != 'healthy'
            if mask.any() and len(healthy_sizes) > 0:
                _, corrected_p_values = multipletests(stats_df.loc[mask, 'pvalue_vs_healthy'], method='fdr_bh')[0:2]
                stats_df.loc[mask, 'pvalue_adjusted'] = corrected_p_values

            if save_plot:
                # Create visualization using the stats DataFrame
                boxes_data = [group_data for _, group_data in stats_df.groupby('group')]
                data = [list(map(float, group['log_sizes'].str.split(';').values[0])) if group['log_sizes'].values[0] else [] for group in boxes_data]
                counts = [len(x) for x in data]
                bp = ax1.boxplot(data,
                                patch_artist=True,
                                showfliers=True,
                                whis=1.5)

                # Add significance markers (using corrected p-values)
                means = stats_df['mean_log10'].tolist()
                pvals = stats_df['pvalue_adjusted'].tolist()
                for i, (mean, p) in enumerate(zip(means, pvals)):
                    if i > 0:  # Skip healthy group
                        marker = ''
                        if p < 0.001:
                            marker = '***'
                        elif p < 0.01:
                            marker = '**'
                        elif p < 0.05:
                            marker = '*'

                        if marker:
                            plt.text(i, mean + 0.2, marker,
                                   ha='center', va='bottom')

                # Customize box appearance
                for box in bp['boxes']:
                    box.set(facecolor=color, alpha=0.7)
                    box.set(linewidth=2)

                # Customize whiskers and caps
                for whisker in bp['whiskers']:
                    whisker.set(linewidth=2)
                for cap in bp['caps']:
                    cap.set(linewidth=2)

                # Customize medians
                for median in bp['medians']:
                    median.set(color='red', linewidth=2)

                # Customize outlier points
                for flier in bp['fliers']:
                    flier.set(marker='o',
                             markerfacecolor='none',
                             markeredgecolor=color,
                             alpha=0.5,
                             markersize=4)

                orig_means = stats_df['mean_raw'].tolist()
                # Plot means on right axis (original scale)
                line = ax2.plot(range(1, len(orig_means) + 1), orig_means,
                               'b-', label='Mean (original scale)',
                               linewidth=2, marker='^', markersize=8)

                # Create a dummy scatter plot for outlier legend
                ax1.scatter([], [], marker='o',
                           facecolors='none',
                           edgecolors=color,
                           alpha=0.5,
                           label='Outliers (>1.5×IQR)')

                # Overlay means
                ax1.plot(range(1, len(means) + 1), means,
                         'r^', label='Mean',
                         markersize=8,
                         markeredgecolor='black')

                # Add sample size annotations
                for i, count in enumerate(counts, 1):
                    ax1.text(i, plt.ylim()[0], f'n={count}',
                            horizontalalignment='center',
                            verticalalignment='top')

                # Customize axes
                ax1.set_xlabel('Disease Severity')
                ax1.set_ylabel('Log10(Clone Size)')
                ax2.set_ylabel('Clone Size (original scale)', color='black')
                ax1.set_ylim(bottom=ax1.get_ylim()[0], top=3.5)  # Set log scale limit

                # Set tick colors
                ax2.tick_params(axis='y', labelcolor='black')
                # Add xticks
                ax1.set_xticks(range(1, len(pattern_names) + 1))
                ax1.set_xticklabels(pattern_names)

                # Combine legends from both axes
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2,
                          loc='upper left',
                          bbox_to_anchor=(1.02, 1.0))

                # Add explanatory text about box plot elements
                plt.figtext(1.02, 0.6,
                            # 'Box Plot Details:\n' +
                            # '• Box: 25-75th percentiles\n' +
                            # '• Red line: median\n' +
                            # '• Whiskers: extend to most\n  extreme non-outlier points\n' +
                            '• Outliers: points beyond\n  1.5×IQR from box edges',
                            bbox=dict(facecolor='white',
                                     alpha=0.8,
                                     edgecolor='none'),
                            transform=plt.gca().transAxes,
                            fontsize=8)

                # Optional: add grid for better readability
                ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

                # Adjust layout to make room for legend
                plt.tight_layout()
                plt.subplots_adjust(right=0.8)  # Make room for the legend

                plt.savefig(f"{outdir}/boxplot_{tcr_type}_k{k}.pdf",
                            bbox_inches='tight', dpi=300)
                plt.close()

            return stats_df

        # Create separate plots and collect statistics
        background_stats = create_comparison_plot('background', 'gray')
        covid_stats = create_comparison_plot('covid', 'red')
        corona_stats = create_comparison_plot('corona', 'orange')

        # Save statistics to separate CSV files
        background_stats.to_csv(f"{outdir}/background_tcr_stats.csv", index=False)
        covid_stats.to_csv(f"{outdir}/covid_tcr_stats.csv", index=False)
        corona_stats.to_csv(f"{outdir}/corona_tcr_stats.csv", index=False)

        return background_stats, covid_stats, corona_stats

    analyze_clone_sizes(df, WOS_PATTERNS, PATTERN_NAMES, k, outdir)

    # Process each severity pattern
    for wos_pattern, pattern_name in zip(WOS_PATTERNS, PATTERN_NAMES):
        print(f"Processing {pattern_name} pattern...")

        # Filter TCRs for current pattern
        pattern_df = df[df['Who Ordinal Scale'].isin(wos_pattern)].copy()
        if sample_data:
            size_before = len(pattern_df)
            pattern_df = pattern_df.sample(n=4497)
            size_after = len(pattern_df)
            print(f"Sample dots for fair comparison. Before: {size_before}, after: {size_after} dots. ")
        tcrs = pattern_df['TRB_1_cdr3'].unique()

        # Calculate BLOSUM distances
        print("Calculating BLOSUM distance matrix...")
        dists = pairwise_blosum62_distance(tcrs, n_proc)

        # Normalize distances and compute UMAP
        scaler = MinMaxScaler()
        dists_normalized = scaler.fit_transform(dists)
        umap = UMAP(metric='precomputed')
        tcr_2d = umap.fit_transform(dists_normalized)

        # Create TCR coordinate mapping
        tcr_coords = {tcr: (x, y) for tcr, (x, y) in zip(tcrs, tcr_2d)}

        # Prepare plot data
        plot_df = pattern_df.copy()
        plot_df['x'] = plot_df['TRB_1_cdr3'].map(lambda x: tcr_coords[x][0])
        plot_df['y'] = plot_df['TRB_1_cdr3'].map(lambda x: tcr_coords[x][1])

        # Determine matched status (COVID-19)
        match_columns = [f'match_{i}' for i in range(k)]
        plot_df['matched'] = plot_df[match_columns].apply(
            lambda row: any(row[col] == 1 for col in match_columns), axis=1)
        # Check association to other coronaviruses
        if consider_corona:
            corona_columns = [f'corona_{i}' for i in range(k)]
            plot_df['corona'] = plot_df[corona_columns].apply(
                lambda row: any(row[col] == 1 for col in corona_columns), axis=1)

        plt.figure(figsize=(12, 10))

        # Get unique clonotype representatives
        def get_unique_clonotypes(df):
            return df.groupby('clonotype').first().reset_index()

        # Plot non-matched TCRs
        non_matched = get_unique_clonotypes(plot_df[~plot_df['matched']])
        plt.scatter(non_matched['x'], non_matched['y'],
                    s=non_matched['clone_size'] * 10,  # Base size multiplier
                    alpha=0.5,
                    color='lightgray',
                    label='Background TCRs')

        # Plot coronavirus-associated TCRs
        if consider_corona:
            corona = get_unique_clonotypes(plot_df[plot_df['corona']])
            plt.scatter(corona['x'], corona['y'],
                        s=corona['clone_size'] * 10,
                        alpha=0.8,
                        color='orange',
                        label='Broad coronavirus-associated TCRs'
            )

        # Plot matched TCRs
        matched = get_unique_clonotypes(plot_df[plot_df['matched']])
        plt.scatter(matched['x'], matched['y'],
                    s=matched['clone_size'] * 10,  # Base size multiplier
                    alpha=0.8,
                    color='red',
                    label='COVID19-associated TCRs')

        # Add legend for clone sizes
        legend_sizes = [2, 10, 20, 30, 50]
        legend_elements = [
            plt.scatter([], [], s=size * 10, color='gray',
                        label=f'{size} TCRs')
            for size in legend_sizes
        ]

        # Add legend
        if legend:
            # First legend for matched status
            plt.legend(loc='upper right')
            # Second legend for clone sizes
            plt.gca().add_artist(plt.legend(
                 handles=legend_elements,
                 title="Clone Sizes",
                 loc='upper left',
                 bbox_to_anchor=(1.15, 1)
            ))

        # Customize plot
        plt.title(f"{pattern_name.capitalize()} Group\n"
                  f"Red: COVID19-associated, \norange: broad coronavirus-associated",
                  fontsize=30)
        plt.xlabel("UMAP 1", fontsize=16)
        plt.ylabel("UMAP 2", fontsize=16)

        # Calculate statistics using unique clonotypes
        unique_plot_df = get_unique_clonotypes(plot_df)
        unique_matched = get_unique_clonotypes(plot_df[plot_df['matched']])

        # Add summary statistics
        stats_text = (
             f"Total unique clonotypes: {len(unique_plot_df)}\n"
             f"Matched unique clonotypes: {len(unique_matched)} "
             f"({len(unique_matched)/len(unique_plot_df)*100:.1f}%)\n"
             f"Mean clone size: {unique_plot_df['clone_size'].mean():.1f}"
        )
        plt.text(0.02, 0.98, stats_text,
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 fontsize=18,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Save plots
        plt.tight_layout()
        # Save as PNG
        plt.savefig(f"{outdir}/tcr_umap_{pattern_name}.png",
                    bbox_inches='tight',
                    dpi=300)
        plt.close()
        print(f"Completed visualization for {pattern_name}")


def expression_level_analysis2(adata, outdir, k=0):
    def create_grouped_gene_expression_plot(genes_of_interest, gene_groups, gene_results, pattern_name, outdir, k):
        """Modified visualization function using log fold changes from DEG analysis"""
        plt.figure(figsize=(15, 10))
        plt.ylim(-2, 2)  # Adjusted for log fold change scale

        # Calculate positions for grouped bars
        group_positions = []
        current_pos = 0

        for group_name in gene_groups.keys():
            group_genes = [g for g in gene_groups[group_name] if g in gene_results]
            if group_genes:
                group_positions.append((group_name, current_pos, current_pos + len(group_genes)))
                current_pos += len(group_genes) + 1.5

        bar_positions = []
        bar_labels = []
        y_offset = 0.1

        for group_name, start_pos, end_pos in group_positions:
            group_genes = [g for g in gene_groups[group_name] if g in gene_results]
            positions = np.arange(start_pos, start_pos + len(group_genes))

            # Plot bars
            for pos, gene in zip(positions, group_genes):
                lfc = gene_results[gene]['logfoldchange']
                color = 'red' if lfc >= 0 else 'blue'
                plt.bar(pos, lfc, color=color, alpha=0.7)

                # Add significance markers
                stars = get_stars(gene_results[gene]['pval_adj'])
                text_y = min(lfc + y_offset, 1.9) if lfc >= 0 else max(lfc - y_offset, -1.9)
                plt.text(pos, text_y, stars, ha='center', va='bottom' if lfc >= 0 else 'top')

            bar_positions.extend(positions)
            bar_labels.extend(group_genes)

            # Add group labels
            group_center = np.mean(positions)
            plt.text(group_center, -2.2, group_name, ha='center', va='top', rotation=90)

        # Customize plot
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        plt.title(f"Differential Gene Expression (Log2 Fold Change PA vs NA)\n{pattern_name}",
                 fontsize=12, pad=20)
        plt.xlabel("Genes", fontsize=10)
        plt.ylabel("Log2 Fold Change", fontsize=10)

        plt.xticks(bar_positions, bar_labels, rotation=90, ha='center')

        # Add legends
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

        plt.savefig(os.path.join(outdir, f"deg_analysis_{pattern_name}_k{k}.pdf"),
                    format='pdf', bbox_inches='tight', dpi=300)
        plt.close()

    print("Stringify Who Ordinal Scale..")
    adata.obs['Who Ordinal Scale'] = adata.obs['Who Ordinal Scale'].apply(clean_wos_value)
    # Preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("Focus on chain_paring == Single pair..")
    adata = adata[adata.obs['chain_pairing'] == 'Single pair']
    os.makedirs(outdir, exist_ok=True)

    # Main axes of analysis remain the same
    wos_patterns = [ALL] + WOS_PATTERNS
    pattern_names = ['all'] + PATTERN_NAMES

    match_columns = [f'match_{i}' for i in range(k)]

    # Main analysis loop
    for wos_pattern, pattern_name in zip(wos_patterns, pattern_names):
        print(f"Processing {pattern_name} group...")

        # Filter data for current severity pattern
        adata_subset = adata[adata.obs['Who Ordinal Scale'].isin(wos_pattern)].copy()

        # Perform DEG analysis
        gene_results = perform_deg_analysis(adata_subset, GENES_OF_INTEREST, match_columns)

        if gene_results:
            # Create visualization
            create_grouped_gene_expression_plot(
                genes_of_interest=GENES_OF_INTEREST,
                gene_groups=GENE_GROUPS,
                gene_results=gene_results,
                pattern_name=pattern_name,
                outdir=outdir,
                k=k
            )


def expression_level_analysis3(adata, outdir, k=0):
    # grouped heatmap plot
    print("Stringify Who Ordinal Scale..")
    adata.obs['Who Ordinal Scale'] = adata.obs['Who Ordinal Scale'].apply(clean_wos_value)

    # Preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print("Focus on chain_paring == Single pair..")
    adata = adata[adata.obs['chain_pairing'] == 'Single pair']

    os.makedirs(outdir, exist_ok=True)

    # Main axes of analysis remain the same
    wos_patterns = [ALL] + WOS_PATTERNS
    pattern_names = ['all'] + PATTERN_NAMES

    match_columns = [f'match_{i}' for i in range(k)]

    def perform_deg_analysis(adata_subset, genes):
        """Perform DEG analysis between PA and NA cells"""
        # Create PA/NA grouping
        adata_subset.obs['comparison_group'] = 'NA'  # Default to NA
        pa_mask = adata_subset.obs[match_columns].apply(
            lambda row: any(row[col] == 1 for col in match_columns), axis=1
        )
        adata_subset.obs.loc[pa_mask, 'comparison_group'] = 'PA'

        # Perform DEG analysis
        try:
            sc.tl.rank_genes_groups(
                adata_subset,
                groupby='comparison_group',
                groups=['PA'],  # Compare PA vs rest (NA)
                reference='NA',
                method='wilcoxon',
                key_added='deg_results',
                pts=True,  # Calculate percentage of cells expressing genes
                genes_batches=genes  # Analyze only genes of interest
            )
        except:
            print("No PA T cells in this group were found, so skip DEG analysis. ")
            return None

        # Extract results
        results = {gene: {} for gene in genes}
        for gene in genes:
            if gene in adata_subset.var_names:
                try:
                    idx = [x[0] for x in adata_subset.uns['deg_results']['names']].index(gene)
                    results[gene] = {
                        'logfoldchange': [x[0] for x in adata_subset.uns['deg_results']['logfoldchanges']][idx],
                        'pval': [x[0] for x in adata_subset.uns['deg_results']['pvals']][idx],
                        'pval_adj': [x[0] for x in adata_subset.uns['deg_results']['pvals_adj']][idx]
                    }
                except:
                    print(f"Cannot find DEG result for the gene: {gene}. Exit program..")
                    breakpoint()
                    exit(0)

        return results

    def create_gene_expression_heatmap(all_patterns_results, gene_groups, outdir, k, pattern_samples, pa_samples):
        """
        Create a heatmap visualization of gene expression across different patterns

        Parameters:
        -----------
        all_patterns_results : dict
            Dictionary with pattern names as keys and gene results as values
        gene_groups : dict
            Dictionary of gene groups and their member genes
        outdir : str
            Output directory path
        k : int
            Parameter k value
        """
        # Create a matrix for the heatmap
        patterns = list(all_patterns_results.keys())
        all_genes = []
        group_boundaries = []
        current_position = 0

        # Collect genes while preserving group order
        for group_name, genes in gene_groups.items():
            valid_genes = [g for g in genes if g in all_patterns_results[patterns[0]]]
            if valid_genes:
                all_genes.extend(valid_genes)
                current_position += len(valid_genes)
                group_boundaries.append((group_name, current_position))

        # Create the data matrix
        data_matrix = np.zeros((len(all_genes), len(patterns)))
        pval_matrix = np.zeros((len(all_genes), len(patterns)))

        # Fill the matrices
        for j, pattern in enumerate(patterns):
            pattern_results = all_patterns_results[pattern]
            for i, gene in enumerate(all_genes):
                if gene in pattern_results:
                    data_matrix[i, j] = pattern_results[gene]['logfoldchange']
                    pval_matrix[i, j] = pattern_results[gene]['pval_adj']

        # Create DataFrame for the heatmap
        df_heatmap = pd.DataFrame(data_matrix,
                                 index=all_genes,
                                 columns=patterns)

        # Create the plot
        plt.figure(figsize=(12, len(all_genes)*0.4 + 2))

        # Modify x-axis labels to include sample sizes
        x_labels = [f"{pattern}\n(N={pattern_samples[pattern]}, PA={pa_samples[pattern]})" for pattern in patterns]

        # Create heatmap
        ax = sns.heatmap(df_heatmap,
                         cmap='RdBu_r',
                         center=0,
                         vmin=-2,
                         vmax=2,
                         annot=True,
                         fmt='.2f',
                         xticklabels=x_labels,
                         cbar_kws={'label': 'Log2 Fold Change'})

        # Add significance asterisks
        for i in range(len(all_genes)):
            for j in range(len(patterns)):
                pval = pval_matrix[i, j]
                if pval <= 0.001:
                    marker = '***'
                elif pval <= 0.01:
                    marker = '**'
                elif pval <= 0.05:
                    marker = '*'
                else:
                    marker = ''
                if marker:
                    ax.text(j + 0.5, i + 0.5, marker,
                           ha='center', va='center',
                           color='black', fontsize=8)

        # Add group labels
        prev_pos = 0
        for group_name, end_pos in group_boundaries:
            middle_pos = prev_pos + (end_pos - prev_pos)/2
            plt.text(-0.5, middle_pos, group_name,
                    ha='right', va='center',
                    fontweight='bold')
            if end_pos < len(all_genes):
                plt.axhline(y=end_pos, color='white', linewidth=2)
            prev_pos = end_pos

        # Customize plot
        plt.title(f"Gene Expression Changes Across Disease Severity\n(PA vs NA)", pad=20)
        plt.xlabel("Disease Severity")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Add legend for significance
        legend_elements = [
            plt.Line2D([0], [0], marker='*', color='w',
                       markerfacecolor='black', markersize=10, label='p ≤ 0.05'),
            plt.Line2D([0], [0], marker='*', color='w',
                       markerfacecolor='black', markersize=15, label='p ≤ 0.01'),
            plt.Line2D([0], [0], marker='*', color='w',
                       markerfacecolor='black', markersize=20, label='p ≤ 0.001')
        ]
        plt.legend(handles=legend_elements,
                  title='Significance Levels',
                  bbox_to_anchor=(1.3, 1),
                  loc='upper right')

        # Save the plot
        plt.savefig(os.path.join(outdir, f"deg_heatmap_k{k}.pdf"),
                    format='pdf', bbox_inches='tight', dpi=300)
        plt.close()

    # The main analysis loop
    all_patterns_results = {}
    # In the main analysis loop, collect sample sizes:
    pattern_samples = {}
    pa_samples = {}
    for wos_pattern, pattern_name in zip(wos_patterns, pattern_names):
        print(f"Processing {pattern_name} group...")
        # Filter data for current severity pattern
        adata_subset = adata[adata.obs['Who Ordinal Scale'].isin(wos_pattern)].copy()

        # Create PA/NA grouping
        adata_subset.obs['comparison_group'] = 'NA'  # Default to NA
        pa_mask = adata_subset.obs[match_columns].apply(
            lambda row: any(row[col] == 1 for col in match_columns), axis=1
        )
        adata_subset.obs.loc[pa_mask, 'comparison_group'] = 'PA'

        # Perform DEG analysis
        gene_results = perform_deg_analysis(adata_subset, GENES_OF_INTEREST)

        # Get total number of samples for this pattern
        pattern_samples[pattern_name] = len(adata_subset)

        # Get number of PA samples
        pa_count = len(adata_subset[adata_subset.obs['comparison_group'] == 'PA'])
        pa_samples[pattern_name] = pa_count

        if gene_results:
            all_patterns_results[pattern_name] = gene_results

    # Create the heatmap
    create_gene_expression_heatmap(
        all_patterns_results=all_patterns_results,
        gene_groups=GENE_GROUPS,
        outdir=outdir,
        k=k,
        pattern_samples=pattern_samples,
        pa_samples=pa_samples
    )


def expression_level_analysis_celltype(adata, outdir, k=0):
    def create_gene_expression_heatmap(all_patterns_results, gene_groups, outdir, k, pattern_samples, pa_samples):
        """
        Create a heatmap visualization of gene expression across different patterns

        Parameters:
        -----------
        all_patterns_results : dict
            Dictionary with pattern names as keys and gene results as values
        gene_groups : dict
            Dictionary of gene groups and their member genes
        outdir : str
            Output directory path
        k : int
            Parameter k value
        """
        # Create a matrix for the heatmap
        patterns = list(all_patterns_results.keys())
        all_genes = []
        group_boundaries = []
        current_position = 0

        # Collect genes while preserving group order
        for group_name, genes in gene_groups.items():
            valid_genes = [g for g in genes if g in all_patterns_results[patterns[0]]]
            if valid_genes:
                all_genes.extend(valid_genes)
                current_position += len(valid_genes)
                group_boundaries.append((group_name, current_position))

        # Create the data matrix
        data_matrix = np.zeros((len(all_genes), len(patterns)))
        pval_matrix = np.zeros((len(all_genes), len(patterns)))

        # Fill the matrices
        for j, pattern in enumerate(patterns):
            pattern_results = all_patterns_results[pattern]
            for i, gene in enumerate(all_genes):
                if gene in pattern_results:
                    data_matrix[i, j] = pattern_results[gene]['logfoldchange']
                    pval_matrix[i, j] = pattern_results[gene]['pval_adj']

        # Create DataFrame for the heatmap
        df_heatmap = pd.DataFrame(data_matrix,
                                 index=all_genes,
                                 columns=patterns)

        # Create the plot
        plt.figure(figsize=(12, len(all_genes)*0.4 + 2))

        # Modify x-axis labels to include sample sizes
        x_labels = [f"{pattern}\n(N={pattern_samples[pattern]}, PA={pa_samples[pattern]})" for pattern in patterns]

        # Create heatmap
        ax = sns.heatmap(df_heatmap,
                         cmap='RdBu_r',
                         center=0,
                         vmin=-2,
                         vmax=2,
                         annot=True,
                         fmt='.2f',
                         xticklabels=x_labels,
                         cbar_kws={'label': 'Log2 Fold Change'})

        # Add significance asterisks
        for i in range(len(all_genes)):
            for j in range(len(patterns)):
                pval = pval_matrix[i, j]
                if pval <= 0.001:
                    marker = '***'
                elif pval <= 0.01:
                    marker = '**'
                elif pval <= 0.05:
                    marker = '*'
                else:
                    marker = ''
                if marker:
                    ax.text(j + 0.5, i + 0.5, marker,
                           ha='center', va='center',
                           color='black', fontsize=8)

        # Add group labels
        prev_pos = 0
        for group_name, end_pos in group_boundaries:
            middle_pos = prev_pos + (end_pos - prev_pos)/2
            plt.text(-0.5, middle_pos, group_name,
                    ha='right', va='center',
                    fontweight='bold')
            if end_pos < len(all_genes):
                plt.axhline(y=end_pos, color='white', linewidth=2)
            prev_pos = end_pos

        # Customize plot
        plt.title(f"Gene Expression Changes Across Disease Severity\n(PA vs NA)", pad=20)
        plt.xlabel("Cell Types")
        plt.xticks(rotation=0, ha='center')
        plt.tight_layout()

        # Add legend for significance
        legend_elements = [
            plt.Line2D([0], [0], marker='*', color='w',
                       markerfacecolor='black', markersize=10, label='p ≤ 0.05'),
            plt.Line2D([0], [0], marker='*', color='w',
                       markerfacecolor='black', markersize=15, label='p ≤ 0.01'),
            plt.Line2D([0], [0], marker='*', color='w',
                       markerfacecolor='black', markersize=20, label='p ≤ 0.001')
        ]
        plt.legend(handles=legend_elements,
                  title='Significance Levels',
                  bbox_to_anchor=(1.3, 1),
                  loc='upper right')

        # Save the plot
        plt.savefig(os.path.join(outdir, f"deg_heatmap_k{k}.pdf"),
                    format='pdf', bbox_inches='tight', dpi=300)
        plt.close()

    # grouped heatmap plot
    print("Stringify Who Ordinal Scale..")
    adata.obs['Who Ordinal Scale'] = adata.obs['Who Ordinal Scale'].apply(clean_wos_value)
    # Preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("Focus on chain_paring == Single pair..")
    adata = adata[adata.obs['chain_pairing'] == 'Single pair']

    os.makedirs(outdir, exist_ok=True)

    match_columns = [f'match_{i}' for i in range(k)]

    # The main analysis loop
    all_patterns_results = {}
    # In the main analysis loop, collect sample sizes:
    pattern_samples = {}
    pa_samples = {}
    for celltype, cell_name in zip(CELL_TYPES, CELL_NAMES):
        print(f"Processing {cell_name} group...")
        # Filter data for current severity pattern
        adata_subset = adata[adata.obs['leiden'].isin(celltype)].copy()

        # Create PA/NA grouping
        adata_subset.obs['comparison_group'] = 'NA'  # Default to NA
        pa_mask = adata_subset.obs[match_columns].apply(
            lambda row: any(row[col] == 1 for col in match_columns), axis=1
        )
        adata_subset.obs.loc[pa_mask, 'comparison_group'] = 'PA'

        # Perform DEG analysis
        gene_results = perform_deg_analysis(adata_subset, GENES_OF_INTEREST, match_columns)

        # Get total number of samples for this pattern
        pattern_samples[cell_name] = len(adata_subset)

        # Get number of PA samples
        pa_count = len(adata_subset[adata_subset.obs['comparison_group'] == 'PA'])
        pa_samples[cell_name] = pa_count

        if gene_results:
            all_patterns_results[cell_name] = gene_results

    # Create the heatmap
    create_gene_expression_heatmap(
        all_patterns_results=all_patterns_results,
        gene_groups=GENE_GROUPS,
        outdir=outdir,
        k=k,
        pattern_samples=pattern_samples,
        pa_samples=pa_samples
    )
