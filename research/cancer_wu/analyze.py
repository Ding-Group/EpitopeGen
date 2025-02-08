import os
import itertools
import pandas as pd
import numpy as np
from scipy import stats
import muon as mu

import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
from pathlib import Path
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import Counter
from matplotlib.patches import Rectangle, Circle

from research.cancer_wu.utils import *


def analysis_wrapper(data_dir, pred_csv, epi_db_path, obs_cache=None, outdir=None, filter_cdr3_notna=True, filter_cell_types=True, top_k=8):
    # A wrapper function to run series of annotations
    mdata = read_all_data(data_dir, obs_cache=obs_cache, filter_cdr3_notna=filter_cdr3_notna, filter_cell_types=filter_cell_types, cdr_annotation_path="cancer_wu/obs_cache/cdr3_added_scirpy.csv")

    PA_ratio_analysis(mdata, outdir=f"{outdir}/PA_ratio_analysis", top_k=top_k)  # Analyze upto top_k
    for k in range(1, 1 + top_k):
        CE_ratio_analysis(mdata, k=k, outdir=f"{outdir}/CE_ratio_analysis")  # k==1 means match_0

    for k in range(1, 1 + top_k):
        antigen_analysis2(mdata, outdir=f"{outdir}/antigen_analysis2", top_k=k)
    antigen_analysis2(mdata, outdir=f"{outdir}/antigen_analysis2", top_k=top_k, color_criterion='site_pattern')

    raw_adata = read_all_raw_data(data_dir)
    raw_adata_filtered = filter_and_update_combined_adata(raw_adata, mdata['gex'])
    # compare_trm_subtypes(mdata, raw_adata_filtered, outdir=f"{outdir}/Trm_subtype_comparison")
    for k in range(1, 1 + top_k):
        expression_level_analysis_grouped(raw_adata_filtered.copy(), outdir=f"{outdir}/gex_grouped", top_k=k)
    for k in range(1, 1 + top_k):
        expression_level_analysis2(raw_adata_filtered.copy(), outdir=f"{outdir}/gex2", top_k=k)
    for k in range(1, 1 + top_k):
        expression_level_analysis(raw_adata_filtered.copy(), outdir=f"{outdir}/gex", top_k=k)

    return mdata


def PA_ratio_analysis(mdata, outdir="analysis/PA_ratio_analysis", top_k=8, per_patient=False):
    """
    Perform Phenotype-Association Ratio analysis that inspects the number of PA T cells
    in a certain repertoire, and visualizes the result as a bar plot.
    """
    df = mdata['gex'].obs
    df = get_cd8_tcells_with_tcrs(df)
    patients = df['patient'].unique()

    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    def create_heatmap(data, title, output_path):
        plt.figure(figsize=(12, 8))
        sns.heatmap(data, annot=True, fmt='.2f', cmap='YlOrRd', vmin=0, vmax=1, cbar_kws={'label': 'PA Ratio'})
        plt.title(title)
        plt.xlabel('Cell Types')
        plt.ylabel('Site Patterns')
        plt.tight_layout()
        plt.savefig(output_path, format='pdf', dpi=300)
        plt.close()

    def create_pa_ratio_bar_plot(data, p_value_df, title, output_path, patterns=None, alpha=0.05):
        """
        Create a bar plot of PA ratios with significance markers comparing to 'All' pattern.

        :param data: pandas DataFrame with PA ratios
        :param p_value_df: pandas DataFrame with corrected p-values
        :param title: str, title of the plot
        :param output_path: str, path to save the output PDF
        :param patterns: list of patterns to include, or None for all patterns
        :param alpha: significance threshold
        """
        if patterns:
            data = data.loc[patterns]
            p_value_df = p_value_df.loc[patterns]

        # Melt the dataframe to long format
        melted_data = data.reset_index().melt(id_vars='index',
                                             var_name='Cell Type',
                                             value_name='PA Ratio')
        melted_data = melted_data.rename(columns={'index': 'Pattern'})

        # Get unique cell types and patterns
        cell_types = melted_data['Cell Type'].unique()
        patterns = melted_data['Pattern'].unique()

        # Set up the plot
        fig, ax = plt.subplots(figsize=(16, 7))
        bar_width = 0.15
        gap_width = 0.02
        group_width = len(patterns) * (bar_width + gap_width) - gap_width

        # Calculate positions for each group of bars
        group_positions = np.arange(len(cell_types))

        # Dictionary to store bar positions for significance markers
        bar_positions_dict = {}

        # Plot bars for each pattern
        for i, pattern in enumerate(patterns):
            pattern_data = melted_data[melted_data['Pattern'] == pattern]
            bar_positions = group_positions + i * (bar_width + gap_width) - group_width / 2 + bar_width / 2
            bars = ax.bar(bar_positions, pattern_data['PA Ratio'],
                         width=bar_width, label=pattern,
                         edgecolor='black', linewidth=1)

            # Store bar positions for significance markers
            bar_positions_dict[pattern] = bar_positions

            # Add significance markers for non-'All' patterns
            if pattern != 'All':
                for j, cell_type in enumerate(cell_types):
                    p_value = p_value_df.loc[pattern, cell_type]
                    if not np.isnan(p_value) and p_value < alpha:
                        # Get heights for current bar and 'All' bar
                        current_height = pattern_data[pattern_data['Cell Type'] == cell_type]['PA Ratio'].values[0]
                        all_height = data.loc['All', cell_type]

                        # Add significance markers
                        if p_value < 0.001:
                            marker = '***'
                        elif p_value < 0.01:
                            marker = '**'
                        elif p_value < 0.05:
                            marker = '*'

                        # Position the marker above the higher bar
                        max_height = max(current_height, all_height)
                        y_pos = max_height + max_height * 0.05  # 5% above the bar

                        ax.text(bar_positions[j], y_pos, marker,
                               ha='center', va='bottom')

        # Customize the plot
        ax.set_xlabel('Cell Types', fontsize=12)
        ax.set_ylabel('PA Ratio', fontsize=12)
        ax.set_title(title)
        ax.set_xticks(group_positions)
        ax.set_xticklabels(cell_types, rotation=45, ha='right')

        # Set y-axis limit with extra space for significance markers
        max_ratio = np.ceil(melted_data['PA Ratio'].max() * 10) / 10
        ax.set_ylim(0, max_ratio * 1.2)  # Add 20% space for markers

        # Add legend
        ax.legend(title='Site Patterns', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add significance level explanation
        ax.text(1.05, 0.5,
                '* p < 0.05\n** p < 0.01\n*** p < 0.001\nCompared to All',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        plt.tight_layout()
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

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

    for n_gen in range(1, 1 + top_k):
        aggregated_data = []
        p_values = []
        num_data = []
        for i, pattern in enumerate(SITE_PATTERNS):
            ratio_row = []
            num_row = []
            p_value_row = []
            for ident in CELL_TYPES:
                # Get data for current pattern and cell type
                df_filtered = df[df['pattern'].isin(pattern) & (df['ident'] == ident)]
                # Get 'All' pattern data for this cell type
                df_all = df[df['ident'] == ident]

                match_cols = [f'match_{i}' for i in range(n_gen)]
                ratio, p_value, _ = calculate_PA_ratio_with_stats(df_filtered, df_all, match_cols)
                ratio_row.append(ratio)
                p_value_row.append(p_value)
                num_row.append(len(df_filtered))

            aggregated_data.append(ratio_row)
            p_values.append(p_value_row)
            num_data.append(num_row)

        # Create DataFrames
        ratio_df = pd.DataFrame(aggregated_data,
                               columns=CELL_TYPES,
                               index=PATTERN_NAMES)
        p_value_df = pd.DataFrame(p_values,
                                 columns=CELL_TYPES,
                                 index=PATTERN_NAMES)
        num_df = pd.DataFrame(num_data,
                              columns=CELL_TYPES,
                              index=PATTERN_NAMES)

        # Apply correction within each cell type
        corrected_p_values = []
        for col in range(p_value_df.shape[1]):
            _, corrected, _, _ = multipletests(p_value_df.iloc[:, col].values, method='fdr_bh')
            corrected_p_values.append(corrected)

        corrected_p_value_df = pd.DataFrame(np.array(corrected_p_values).T,
                                           columns=CELL_TYPES,
                                           index=PATTERN_NAMES)

        # Rename the index
        for k, v in PATTERN_NAMES2DESC.items():
            ratio_df = ratio_df.rename(index=PATTERN_NAMES2DESC)
            corrected_p_value_df = corrected_p_value_df.rename(index=PATTERN_NAMES2DESC)

        # create_heatmap(ratio_df, 'PA Ratio - All Patients', os.path.join(outdir, 'PA_ratio_heatmap_all_patients.pdf'))
        subgroup1 = ['All', 'Tumor Singleton (TS)', 'Tumor Multiplet (TM)', 'Dual Expansion (DE)']
        subgroup2 = ['All', 'Dual Expansion, Blood Independent (DEBI)',
                     'Dual Expansion, Blood Non-expanded (DEBN)', 'Dual Expansion, Blood Expanded (DEBE)']  # DE subtypes

        # Save data to CSV
        ratio_df.to_csv(f'{outdir}/PA_ratio_all_patients_k{n_gen}.csv')
        corrected_p_value_df.to_csv(f'{outdir}/p_value_all_patients_k{n_gen}.csv')
        num_df.to_csv(f"{outdir}/num_cells_all_patients_k{n_gen}.csv")
        create_pa_ratio_bar_plot(ratio_df,
                                corrected_p_value_df,  # Use corrected p-values
                                'PA Ratio - Tumor and DE Patterns',
                                os.path.join(outdir, f'PA_ratio_k{n_gen}.pdf'),
                                patterns=subgroup1)

        create_pa_ratio_bar_plot(ratio_df,
                                corrected_p_value_df,  # Use corrected p-values
                                'PA Ratio - DE and Blood Expansion Patterns',
                                os.path.join(outdir, f'PA_ratio_DE_subgroup_k{n_gen}.pdf'),
                                patterns=subgroup2)

    # For each patient
    if per_patient:
        patient_ratios = {}
        patient_pvalues = {}

        for patient in patients:
            patient_data = []
            patient_pvals = []

            for i, pattern in enumerate(SITE_PATTERNS):
                ratio_row = []
                pval_row = []

                for ident in CELL_TYPES:
                    # Get data for current pattern and cell type
                    df_filtered = df[(df['pattern'].isin(pattern)) &
                                   (df['ident'] == ident) &
                                   (df['patient'] == patient)]

                    # Get 'All' pattern data for this patient and cell type
                    df_all = df[(df['ident'] == ident) &
                               (df['patient'] == patient)]

                    ratio, p_value, _ = calculate_PA_ratio_with_stats(df_filtered, df_all)
                    ratio_row.append(ratio)
                    pval_row.append(p_value)

                patient_data.append(ratio_row)
                patient_pvals.append(pval_row)

            # Create DataFrames for this patient
            patient_df = pd.DataFrame(patient_data,
                                    columns=CELL_TYPES,
                                    index=PATTERN_NAMES)
            pvalue_df = pd.DataFrame(patient_pvals,
                                   columns=CELL_TYPES,
                                   index=PATTERN_NAMES)

            # Apply multiple testing correction within each cell type for this patient
            corrected_pvals = []
            for col in range(pvalue_df.shape[1]):
                col_pvals = pvalue_df.iloc[:, col]
                # Remove NaN values before correction
                mask = ~np.isnan(col_pvals)
                if mask.sum() > 0:  # If there are any non-NaN values
                    _, corrected, _, _ = multipletests(col_pvals[mask], method='fdr_bh')
                    corrected_col = np.full(len(col_pvals), np.nan)
                    corrected_col[mask] = corrected
                    corrected_pvals.append(corrected_col)
                else:
                    corrected_pvals.append(np.full(len(col_pvals), np.nan))

            corrected_pvalue_df = pd.DataFrame(np.array(corrected_pvals).T,
                                             columns=CELL_TYPES,
                                             index=PATTERN_NAMES)
            for k, v in PATTERN_NAMES2DESC.items():
                patient_df = patient_df.rename(index=PATTERN_NAMES2DESC)
                corrected_pvalue_df = corrected_pvalue_df.rename(index=PATTERN_NAMES2DESC)

            # create_heatmap(patient_df, f'PA Ratio - Patient {patient}', os.path.join(outdir, f'PA_ratio_heatmap_patient_{patient}.pdf'))
            create_pa_ratio_bar_plot(patient_df,
                                    corrected_pvalue_df,  # Use corrected p-values
                                    f'PA Ratio - Tumor and DE Patterns: {patient}',
                                    os.path.join(outdir, f'PA_ratio_tumor_DE_{patient}.pdf'),
                                    patterns=subgroup1)

            create_pa_ratio_bar_plot(patient_df,
                                    corrected_pvalue_df,  # Use corrected p-values
                                    f'PA Ratio - DE and Blood Expansion Patterns: {patient}',
                                    os.path.join(outdir, f'PA_ratio_DE_blood_expansion_{patient}.pdf'),
                                    patterns=subgroup2)



            patient_df = pd.DataFrame(patient_data, columns=CELL_TYPES, index=PATTERN_NAMES)
            patient_df.to_csv(os.path.join(outdir, f'PA_ratio_patient_{patient}.csv'))
            corrected_pvalue_df.to_csv(os.path.join(outdir, f'p_value_patient_{patient}.csv'))

    print(f"PA ratio analysis completed. Results saved in {outdir}")
    return mdata


def CE_ratio_analysis(mdata, k=1, outdir="analysis/240828_CE_ratio_analysis", per_patient=False):
    df = mdata['gex'].obs
    df = get_cd8_tcells_with_tcrs(df)
    patients = df['patient'].unique()

    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    def calculate_clone_sizes(data):
        return data.groupby('cdr3_nt').size()

    def monte_carlo_clone_comparison(PA_clones, NA_clones, n_iterations=1000):
        observed_ratio = np.mean(PA_clones) / np.mean(NA_clones)
        combined = np.concatenate([PA_clones, NA_clones])
        n, m = len(PA_clones), len(NA_clones)

        count_more_extreme = 0
        for _ in range(n_iterations):
            np.random.shuffle(combined)
            sample_PA = combined[:n]
            sample_NA = combined[n:]
            sample_ratio = np.mean(sample_PA) / np.mean(sample_NA)
            if sample_ratio >= observed_ratio:
                count_more_extreme += 1

        p_value = count_more_extreme / n_iterations
        return observed_ratio, p_value

    def compare_clone_sizes(data, k):
        """
        Compare clone sizes between PA (any match) and NA (all non-matches) groups based on k matching columns.

        Args:
            data: DataFrame containing match_0 to match_{k-1} columns and clone size information
            k: Number of match columns to consider

        Returns:
            Tuple of (mean_PA, mean_NA, ratio, p_value)
        """
        # Verify all required match columns exist
        match_cols = [f'match_{i}' for i in range(k)]
        if not all(col in data.columns for col in match_cols):
            raise ValueError(f"Data missing one or more required match columns for k={k}")

        # Create PA mask (any match is true - OR condition)
        PA_mask = data[match_cols].any(axis=1)

        # Create NA mask (all matches are false - AND condition with negation)
        NA_mask = ~data[match_cols].any(axis=1)

        # Calculate clone sizes for each group
        PA_clones = calculate_clone_sizes(data[PA_mask])
        NA_clones = calculate_clone_sizes(data[NA_mask])

        # Return NaN values if either group is empty
        if len(PA_clones) == 0 or len(NA_clones) == 0:
            return np.nan, np.nan, np.nan, np.nan, len(PA_clones), len(NA_clones)

        # Calculate means
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

    def create_heatmap(data, title, output_path, cmap='YlOrRd', dpi=300):
        plt.figure(figsize=(12, 8))
        sns.heatmap(data, annot=True, fmt='.2f', cmap=cmap, cbar_kws={'label': ''})
        plt.title(title)
        plt.xlabel('Cell Types')
        plt.ylabel('Site Patterns')
        plt.tight_layout()
        plt.savefig(output_path, format='pdf', dpi=dpi)
        plt.close()

    def create_clone_size_ratio_plot(mean_PA_df, p_value_df, ratio_df, n_PA_df, n_NA_df, title, output_path, debug=False, use_DE_subgroup=False):
        """
        Create a plot with circles representing clone size and color representing ratio.

        :param mean_PA_df: DataFrame with mean PA clone sizes
        :param ratio_df: DataFrame with PA/NA ratios
        :param title: str, title of the plot
        :param output_path: str, path to save the output PDF
        """
        if not use_DE_subgroup:
            print("Exclude DE subgroup information..")
            mean_PA_df = mean_PA_df[~mean_PA_df.index.isin(["DE, Blood Independent", "DE, Blood Non-expanded", "DE, Blood-expanded"])]
            p_value_df = p_value_df[~p_value_df.index.isin(["DE, Blood Independent", "DE, Blood Non-expanded", "DE, Blood-expanded"])]
            ratio_df = ratio_df[~ratio_df.index.isin(["DE, Blood Independent", "DE, Blood Non-expanded", "DE, Blood-expanded"])]
            n_PA_df = n_PA_df[~n_PA_df.index.isin(["DE, Blood Independent", "DE, Blood Non-expanded", "DE, Blood-expanded"])]
            n_NA_df = n_NA_df[~n_NA_df.index.isin(["DE, Blood Independent", "DE, Blood Non-expanded", "DE, Blood-expanded"])]

        # Calculate figure size based on number of rows and columns
        n_rows, n_cols = mean_PA_df.shape
        cell_size = 0.9  # Size of each cell in inches
        fig_width = n_cols * cell_size + 2  # Add extra space for labels and colorbar
        fig_height = n_rows * cell_size*0.9  # Add extra space for title and labels

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        # plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)  # Adjust these values as needed

        # Define color map
        cmap = plt.cm.RdBu_r
        norm = plt.Normalize(vmin=0.0, vmax=2.0)  # Assuming ratios around 1 are neutral

        # Calculate circle sizes
        sizes = mean_PA_df
        for i, pattern in enumerate(mean_PA_df.index):
            for j, cell_type in enumerate(mean_PA_df.columns):
                size = min(0.35, sizes.iloc[i, j] / 30)
                ratio = ratio_df.iloc[i, j]
                color = cmap(norm(ratio))

                circle = plt.Circle((j + 0.5, i + 0.5), size, facecolor=color, alpha=0.7, linewidth=1, edgecolor='black')
                ax.add_artist(circle)
                if p_value_df.iloc[i, j] < 0.05 or debug:
                    ax.text(j + 0.35, i + 0.75, f"pval={p_value_df.iloc[i, j]:.3f}", fontsize=5)
                # Add text for clone size
                if size >= 6 / 30:
                    if debug:
                        ax.text(j + 0.5, i + 0.5, f'{mean_PA_df.iloc[i, j]:.2f}, {n_PA_df.iloc[i, j]}:{n_NA_df.iloc[i, j]}', ha='center', va='center', fontsize=7)
                    else:
                        ax.text(j + 0.5, i + 0.5, f'{mean_PA_df.iloc[i, j]:.2f}', ha='center', va='center', fontsize=7)
                else:
                    if debug:
                        ax.text(j + 0.7, i + 0.3, f'{mean_PA_df.iloc[i, j]:.2f}, {n_PA_df.iloc[i, j]}:{n_NA_df.iloc[i, j]}', ha='center', va='center', fontsize=7)
                    else:
                        ax.text(j + 0.7, i + 0.3, f'{mean_PA_df.iloc[i, j]:.2f}', ha='center', va='center', fontsize=7)

        # Set labels and title
        ax.set_xlim(0, len(mean_PA_df.columns))
        ax.set_ylim(0, len(mean_PA_df.index))
        ax.set_xticks(np.arange(len(mean_PA_df.columns)) + 0.5)
        ax.set_yticks(np.arange(len(mean_PA_df.index)) + 0.5)
        ax.set_xticklabels(mean_PA_df.columns, rotation=45, ha='right', fontdict={'fontsize': 7})
        ax.set_yticklabels(mean_PA_df.index, fontdict={'fontsize': 7})
        # ax.set_xlabel('Cell Types')
        # ax.set_ylabel('Site Patterns')
        # ax.set_title(title)

        # Invert y-axis to match DataFrame orientation
        ax.invert_yaxis()

        # Add color bar for ratio
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label('avg clone size (PA) / avg clone size (NA)')
        for t in cbar.ax.get_yticklabels():
             t.set_fontsize(7)

        plt.tight_layout()
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

    # All patients aggregated
    mean_PA_sizes = []
    mean_NA_sizes = []
    ratio_sizes = []
    p_values = []
    n_PA_sizes = []
    n_NA_sizes = []

    for i, pattern in enumerate(SITE_PATTERNS):
        mean_PA_row = []
        mean_NA_row = []
        ratio_row = []
        p_value_row = []
        n_PA_row = []
        n_NA_row = []
        for ident in CELL_TYPES:
            df_filtered = df[(df['pattern'].isin(pattern)) & (df['ident'] == ident)]
            mean_PA, mean_NA, ratio, p_value, n_PA, n_NA = compare_clone_sizes(df_filtered, k=k)
            mean_PA_row.append(mean_PA)
            mean_NA_row.append(mean_NA)
            ratio_row.append(ratio)
            p_value_row.append(p_value)
            n_PA_row.append(n_PA)
            n_NA_row.append(n_NA)

        mean_PA_sizes.append(mean_PA_row)
        mean_NA_sizes.append(mean_NA_row)
        ratio_sizes.append(ratio_row)
        p_values.append(p_value_row)
        n_PA_sizes.append(n_PA_row)
        n_NA_sizes.append(n_NA_row)

    # Multiple test correction of the p values
    flat_p_values = np.array(p_values).flatten()
    # Apply BH correction
    rejected, corrected_p_values, _, _ = multipletests(flat_p_values,
                                                     method='fdr_bh')
    corrected_p_values = corrected_p_values.reshape(len(SITE_PATTERNS), len(CELL_TYPES))

    mean_PA_df = pd.DataFrame(mean_PA_sizes, columns=CELL_TYPES, index=PATTERN_NAMES)
    mean_NA_df = pd.DataFrame(mean_NA_sizes, columns=CELL_TYPES, index=PATTERN_NAMES)
    ratio_df = pd.DataFrame(ratio_sizes, columns=CELL_TYPES, index=PATTERN_NAMES)
    p_value_df = pd.DataFrame(corrected_p_values, columns=CELL_TYPES, index=PATTERN_NAMES)
    n_PA_df = pd.DataFrame(n_PA_sizes, columns=CELL_TYPES, index=PATTERN_NAMES)
    n_NA_df = pd.DataFrame(n_NA_sizes, columns=CELL_TYPES, index=PATTERN_NAMES)

    create_clone_size_ratio_plot(mean_PA_df, p_value_df, ratio_df, n_PA_df, n_NA_df, 'Clone Size and PA/NA Ratio - All Patients',
                                 os.path.join(outdir, f'CE_Ratio_all_patients_k{k}.pdf'))
    # Save aggregated data to CSV
    mean_PA_df.to_csv(os.path.join(outdir, f'mean_PA_clone_sizes_all_patients_k{k}.csv'))
    p_value_df.to_csv(os.path.join(outdir, f'clone_size_comparison_p_values_all_patients_k{k}.csv'))

    if per_patient:
        # For each patient
        for patient in patients:
            patient_mean_PA_sizes = []
            patient_mean_NA_sizes = []
            patient_ratio_sizes = []
            patient_p_values = []
            patient_n_PA_sizes = []
            patient_n_NA_sizes = []

            for i, pattern in enumerate(SITE_PATTERNS):
                mean_PA_row = []
                mean_NA_row = []
                ratio_row = []
                p_value_row = []
                n_PA_row = []
                n_NA_row = []
                for ident in CELL_TYPES:
                    df_filtered = df[(df['pattern'].isin(pattern)) & (df['ident'] == ident) & (df['patient'] == patient)]
                    mean_PA, mean_NA, ratio, p_value, n_PA, n_NA = compare_clone_sizes(df_filtered, k=k)
                    mean_PA_row.append(mean_PA)
                    mean_NA_row.append(mean_NA)
                    ratio_row.append(ratio)
                    p_value_row.append(p_value)
                    n_PA_row.append(n_PA)
                    n_NA_row.append(n_NA)
                patient_mean_PA_sizes.append(mean_PA_row)
                patient_mean_NA_sizes.append(mean_NA_row)
                patient_ratio_sizes.append(ratio_row)
                patient_p_values.append(p_value_row)
                patient_n_PA_sizes.append(n_PA_row)
                patient_n_NA_sizes.append(n_NA_row)

            # Multiple test correction
            flat_p_values = np.array(patient_p_values).flatten()
            # Apply BH correction
            rejected, corrected_p_values, _, _ = multipletests(flat_p_values,
                                                             method='fdr_bh')
            corrected_p_values = corrected_p_values.reshape(len(SITE_PATTERNS), len(CELL_TYPES))

            patient_mean_PA_df = pd.DataFrame(patient_mean_PA_sizes, columns=CELL_TYPES, index=PATTERN_NAMES)
            patient_mean_NA_df = pd.DataFrame(patient_mean_NA_sizes, columns=CELL_TYPES, index=PATTERN_NAMES)
            patient_ratio_df = pd.DataFrame(patient_ratio_sizes, columns=CELL_TYPES, index=PATTERN_NAMES)
            patient_p_value_df = pd.DataFrame(corrected_p_values, columns=CELL_TYPES, index=PATTERN_NAMES)
            patient_n_PA_df = pd.DataFrame(patient_n_PA_sizes, columns=CELL_TYPES, index=PATTERN_NAMES)
            patient_n_NA_df = pd.DataFrame(patient_n_NA_sizes, columns=CELL_TYPES, index=PATTERN_NAMES)

            create_clone_size_ratio_plot(patient_mean_PA_df, patient_p_value_df, patient_ratio_df, patient_n_PA_df, patient_n_NA_df,
                                         f'Clone Size and PA/NA Ratio - {patient}',
                                         os.path.join(outdir, f'CE_Ratio_{patient}_k{k}.pdf'))

            # Save data to CSV
            patient_mean_PA_df.to_csv(os.path.join(outdir, f'mean_PA_clone_sizes_patient_{patient}_k{k}.csv'))
            patient_p_value_df.to_csv(os.path.join(outdir, f'clone_size_comparison_p_values_patient_{patient}_k{k}.csv'))

    print(f"Clone size comparison analysis completed. Results saved in {outdir}")


def create_volcano_plot(results, title, output_path):
    plt.figure(figsize=(10, 8))

    # Calculate -log10(p-value) and log2(fold change)
    log10_p = -np.log10(results['pvals_adj'])
    log2_fc = np.log2(results['logfoldchanges'])

    # Plot scatter
    plt.scatter(log2_fc, log10_p, alpha=0.5)

    # Add labels for top genes
    top_genes = results.sort_values('pvals_adj').head(10)
    for _, gene in top_genes.iterrows():
        plt.annotate(gene['names'],
                     (np.log2(gene['logfoldchanges']), -np.log10(gene['pvals_adj'])),
                     xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Add lines for fold change and p-value thresholds
    plt.axhline(y=-np.log10(0.05), color='r', linestyle='--')
    plt.axvline(x=1, color='r', linestyle='--')
    plt.axvline(x=-1, color='r', linestyle='--')

    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 Adjusted p-value')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close()


def compare_trm_subtypes(mdata, filtered_raw_adata, outdir="analysis/trm_subtype_comparison"):
    # Measure the expression levels of some genes of interest
    # No need to select HVGs in this case

    adata = mdata['gex']
    sc.pp.normalize_total(filtered_raw_adata, target_sum=1e4)
    sc.pp.log1p(filtered_raw_adata)

    trm_subtypes = ['8.3a-Trm', '8.3b-Trm', '8.3c-Trm']

    os.makedirs(outdir, exist_ok=True)

    # 1. Compare expression levels of specific genes
    genes_of_interest = ['PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT']  # Add more genes as needed

    for gene in genes_of_interest:
        if gene not in filtered_raw_adata.var_names:
            print(f"Warning: Gene {gene} not found in the dataset.")
            continue

        # Create a DataFrame for plotting
        gene_index = filtered_raw_adata.var_names.get_loc(gene)
        gene_expression = filtered_raw_adata.X[:, gene_index]
        plot_data = pd.DataFrame({
            'Cell Type': filtered_raw_adata.obs['ident'],
            'Expression': np.array(gene_expression.todense())[:, 0]
        })
        plot_data = plot_data[plot_data['Cell Type'].isin(trm_subtypes)]

        # Calculate and print the mean expression for each cell type
        mean_expression_by_type = plot_data.groupby('Cell Type')['Expression'].mean()
        print(f"Mean expression of {gene} by cell type:")
        print(mean_expression_by_type)

        # Plotting bar plot of mean expression
        plt.figure(figsize=(10, 6))
        sns.barplot(x=mean_expression_by_type.index, y=mean_expression_by_type.values, palette="viridis")
        plt.title(f"Mean {gene} Expression in TRM Subtypes")
        plt.ylabel(f"Mean {gene} Expression")
        plt.xlabel("Cell Type")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{gene}_mean_expression_trm_subtypes.pdf"), format='pdf')
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Cell Type', y='Expression', data=plot_data, inner='box')
        sns.stripplot(x='Cell Type', y='Expression', data=plot_data, color='black', size=2, alpha=0.4)
        plt.title(f"{gene} Expression in TRM Subtypes")
        plt.ylabel(f"{gene} Expression")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{gene}_expression_trm_subtypes.pdf"), format='pdf')
        plt.close()

    # 2. Perform differential expression analysis among TRM subtypes
    adata_trm = filtered_raw_adata[filtered_raw_adata.obs['ident'].isin(trm_subtypes)].copy()

    sc.tl.rank_genes_groups(adata_trm, groupby='ident', method='wilcoxon')

    # Dictionary to store top DEGs for each subtype
    top_degs = {}

    # Generate and save results
    for subtype in trm_subtypes:
        results = sc.get.rank_genes_groups_df(adata_trm, group=subtype)
        results.to_csv(os.path.join(outdir, f"DEGs_{subtype}_vs_other_TRMs.csv"))

        # Store top 20 genes for heatmap
        top_degs[subtype] = results.head(20)['names'].tolist()

        # Create volcano plot
        create_volcano_plot(results, f"Volcano Plot - {subtype} vs Other TRM Subtypes", os.path.join(outdir, f"volcano_plot_{subtype}_vs_other_TRMs.pdf"))

    # 3. Heatmap of top differentially expressed genes
    # Combine all top DEGs
    all_top_genes = list(set([gene for genes in top_degs.values() for gene in genes]))

    # Create a DataFrame for the heatmap
    heatmap_data = pd.DataFrame(index=all_top_genes, columns=trm_subtypes)

    for subtype in trm_subtypes:
        subtype_data = adata_trm[adata_trm.obs['ident'] == subtype]
        mean_expr = subtype_data[:, all_top_genes].X.mean(axis=0)
        heatmap_data[subtype] = np.array(mean_expr)[0]

    # Plot heatmap
    plt.figure(figsize=(12, 20))
    sns.heatmap(heatmap_data, cmap='YlOrRd', center=0, yticklabels=True)
    plt.title("Top DEGs Across TRM Subtypes")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "top_degs_heatmap.pdf"), format='pdf')
    plt.close()

    # Plot clustered heatmap
    plt.figure(figsize=(12, 20))
    sns.clustermap(heatmap_data, cmap='YlOrRd', center=0, yticklabels=True, figsize=(12, 20))
    plt.suptitle("Clustered Top DEGs Across TRM Subtypes", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "top_degs_clustered_heatmap.pdf"), format='pdf')
    plt.close()

    # Bar plot for top genes in each TRM subtype
    for subtype in trm_subtypes:
        top_10_genes = top_degs[subtype][:10]
        gene_expr = adata_trm[adata_trm.obs['ident'] == subtype, top_10_genes].X.mean(axis=0)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_10_genes, y=np.array(gene_expr)[0])
        plt.title(f"Top 10 DEGs in {subtype}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"top_10_degs_barplot_{subtype}.pdf"), format='pdf')
        plt.close()

    print(f"TRM subtype comparison completed. Results saved in {outdir}")


def expression_level_analysis(filtered_raw_adata, outdir, top_k=1):
    """
    Analyze gene expression differences across cell types and phenotypes.

    Parameters
    ----------
    filtered_raw_adata: AnnData object
        containing gene expression data
    outdir: str
        Output directory for saving results
    top_k: int
        How many number of generations per TCR to consider
    """
    def perform_deg_analysis(adata_subset, genes_of_interest, p_values, fold_changes):
        # Initialize storage for results
        try:
            sc.tl.rank_genes_groups(adata_subset,
                                   groupby='comparison_group',
                                   groups=['PA'],
                                   reference='NA',
                                   method='wilcoxon',
                                   pts=True)

            # Extract results for genes of interest
            try:
                de_results = sc.get.rank_genes_groups_df(adata_subset, group='PA')
            except Exception as e:
                print(f"Warning: Could not get rank genes groups df: {str(e)}")
                de_results = pd.DataFrame()  # Empty DataFrame as fallback

            for gene in genes_of_interest:
                try:
                    # Get results for this gene
                    gene_results = de_results[de_results['names'] == gene]

                    if len(gene_results) > 0:
                        try:
                            p_values[gene].append(gene_results['pvals'].iloc[0])
                            fold_changes[gene].append(gene_results['logfoldchanges'].iloc[0])
                        except Exception as e:
                            print(f"Warning: Error processing gene {gene}: {str(e)}")
                            p_values[gene].append(np.nan)
                            fold_changes[gene].append(np.nan)
                    else:
                        print(f"Warning: No DE results found for gene {gene}")
                        p_values[gene].append(np.nan)
                        fold_changes[gene].append(np.nan)

                except Exception as e:
                    print(f"Warning: Error processing gene {gene}: {str(e)}")
                    p_values[gene] = [np.nan]
                    fold_changes[gene] = [np.nan]

        except Exception as e:
            print(f"Warning: Error in rank_genes_groups analysis: {str(e)}")
            # Initialize all genes with NaN values
            for gene in genes_of_interest:
                p_values[gene] = [np.nan]
                fold_changes[gene] = [np.nan]
        return p_values, fold_changes

    def create_gene_expression_heatmap(fold_changes, p_values, gene_groups, pattern_names, outdir, k):
        # Initialize lists to store genes and track group boundaries
        all_genes = []
        group_boundaries = []
        current_position = 0

        # Collect valid genes from each group and track boundaries
        for group_name, genes in gene_groups.items():
            valid_genes = [gene for gene in genes if gene in fold_changes]
            if valid_genes:
                # Sort genes within group by average fold change
                avg_fold_changes = {gene: np.mean(fold_changes[gene]) for gene in valid_genes}
                sorted_group_genes = sorted(valid_genes, key=lambda g: avg_fold_changes[g], reverse=True)

                all_genes.extend(sorted_group_genes)
                current_position += len(sorted_group_genes)
                group_boundaries.append((group_name, current_position))

        # Create heatmap data array
        heatmap_data = np.array([fold_changes[gene] for gene in all_genes])

        # Create figure with adjusted height to accommodate groups
        plt.figure(figsize=(12, len(all_genes) * 0.4 + 1))

        # Create main heatmap
        sns.heatmap(heatmap_data,
                    annot=True,
                    fmt=".2f",
                    cmap="RdBu_r",
                    center=0,
                    xticklabels=pattern_names,
                    yticklabels=all_genes,
                    cbar_kws={'label': 'Log2 Fold Change (PA/NA)'})

        # Add significance asterisks
        for i, gene in enumerate(all_genes):
            for j in range(len(pattern_names)):
                if p_values[gene][j] < 0.001:
                    plt.text(j + 0.7, i + 0.5, '***', ha='center', va='center')
                elif p_values[gene][j] < 0.01:
                    plt.text(j + 0.7, i + 0.5, '**', ha='center', va='center')
                elif p_values[gene][j] < 0.05:
                    plt.text(j + 0.7, i + 0.5, '*', ha='center', va='center')

        # Add group labels and boundaries
        prev_pos = 0
        for group_name, end_pos in group_boundaries:
            middle_pos = prev_pos + (end_pos - prev_pos)/2
            plt.text(-0.5, middle_pos, group_name,
                    ha='right', va='center',
                    fontweight='bold')
            if end_pos < len(all_genes):
                plt.axhline(y=end_pos, color='white', linewidth=2)
            prev_pos = end_pos

        # Adjust layout and labels
        plt.title('Differential Gene Expression Between PA and NA Across Cell Types')
        plt.xlabel('Cell Type')
        plt.ylabel('Genes')

        # Save plots with tight layout
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"gex_heatmap_celltype_k{k}.pdf"), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(outdir, f"gex_heatmap_celltype_k{k}.png"), format='png', dpi=300, bbox_inches='tight')
        plt.close()

        # Save numerical results
        results_df = pd.DataFrame({
            'Gene': all_genes,
            'Group': [next(group_name for group_name, end_pos in group_boundaries if end_pos > i)
                     for i in range(len(all_genes))],
            **{f"{pattern}_fold_change": [fold_changes[gene][i] for gene in all_genes]
               for i, pattern in enumerate(pattern_names)},
            **{f"{pattern}_pvalue": [p_values[gene][i] for gene in all_genes]
               for i, pattern in enumerate(pattern_names)}
        })
        results_df.to_csv(os.path.join(outdir, f"gex_heatmap_celltype_k{k}.csv"), index=False)

        return results_df

    sc.pp.normalize_total(filtered_raw_adata, target_sum=1e4)
    sc.pp.log1p(filtered_raw_adata)
    os.makedirs(outdir, exist_ok=True)

    p_values = {gene: [] for gene in GENES_OF_INTEREST}  # dict of list
    fold_changes = {gene: [] for gene in GENES_OF_INTEREST}  # dict of list
    pattern_sizes = {}
    pattern_pa_sizes = {}
    # main loop: process each pattern
    for cell_type in CELL_TYPES:
        # Filter data for current pattern
        pattern_adata = filtered_raw_adata[filtered_raw_adata.obs['ident'] == cell_type].copy()

        # Create a temporary grouping column for DE analysis
        pattern_adata.obs['comparison_group'] = 'NA'
        pa_mask = pd.Series(False, index=pattern_adata.obs.index)
        for k in range(top_k):
            match_col = f'match_{k}'
            if match_col in pattern_adata.obs.columns:
                pa_mask |= (pattern_adata.obs[match_col] == 1)
        # Assign PA label to cells with any match in top K
        pattern_adata.obs.loc[pa_mask, 'comparison_group'] = 'PA'
        # Print statistics about the grouping
        n_pa = sum(pattern_adata.obs['comparison_group'] == 'PA')
        n_total = len(pattern_adata.obs)
        print(f"Num of PA cells (top {top_k} matches): {n_pa} ({n_pa/n_total*100:.2f}%) out of total {n_total} cells")

        pattern_sizes[cell_type] = n_total
        pattern_pa_sizes[cell_type] = n_pa

        # Perform differential gene expression analysis
        p_values, fold_changes = perform_deg_analysis(pattern_adata, GENES_OF_INTEREST, p_values, fold_changes)

    pattern_names_with_counts = [
        f"{cell_type}\n(Total N={pattern_sizes[cell_type]:,}\nPA N={pattern_pa_sizes[cell_type]:,}, {pattern_pa_sizes[cell_type]/pattern_sizes[cell_type]*100:.1f}%)"
        for cell_type in CELL_TYPES
    ]

    results_df = create_gene_expression_heatmap(fold_changes, p_values, GENE_GROUPS, pattern_names_with_counts, outdir, top_k)

    return results_df


def expression_level_analysis2(filtered_raw_adata, outdir, use_scanpy_de=True, top_k=1):
    """
    Analyze gene expression differences between PA and NA T cells across different expansion patterns.

    Args:
        filtered_raw_adata: AnnData object containing gene expression data
        outdir: Output directory for saving results
    """
    def perform_deg_analysis(adata_subset, genes_of_interest, p_values, fold_changes):
        # Initialize storage for results
        try:
            sc.tl.rank_genes_groups(adata_subset,
                                   groupby='comparison_group',
                                   groups=['PA'],
                                   reference='NA',
                                   method='wilcoxon',
                                   pts=True)

            # Extract results for genes of interest
            try:
                de_results = sc.get.rank_genes_groups_df(adata_subset, group='PA')
            except Exception as e:
                print(f"Warning: Could not get rank genes groups df: {str(e)}")
                de_results = pd.DataFrame()  # Empty DataFrame as fallback

            for gene in genes_of_interest:
                try:
                    # Get results for this gene
                    gene_results = de_results[de_results['names'] == gene]

                    if len(gene_results) > 0:
                        try:
                            p_values[gene].append(gene_results['pvals'].iloc[0])
                            fold_changes[gene].append(gene_results['logfoldchanges'].iloc[0])
                        except Exception as e:
                            print(f"Warning: Error processing gene {gene}: {str(e)}")
                            p_values[gene].append(np.nan)
                            fold_changes[gene].append(np.nan)
                    else:
                        print(f"Warning: No DE results found for gene {gene}")
                        p_values[gene].append(np.nan)
                        fold_changes[gene].append(np.nan)

                except Exception as e:
                    print(f"Warning: Error processing gene {gene}: {str(e)}")
                    p_values[gene] = [np.nan]
                    fold_changes[gene] = [np.nan]

        except Exception as e:
            print(f"Warning: Error in rank_genes_groups analysis: {str(e)}")
            # Initialize all genes with NaN values
            for gene in genes_of_interest:
                p_values[gene] = [np.nan]
                fold_changes[gene] = [np.nan]
        return p_values, fold_changes

    def create_gene_expression_heatmap(fold_changes, p_values, gene_groups, pattern_names, outdir, k):
        # Initialize lists to store genes and track group boundaries
        all_genes = []
        group_boundaries = []
        current_position = 0

        # Collect valid genes from each group and track boundaries
        for group_name, genes in gene_groups.items():
            valid_genes = [gene for gene in genes if gene in fold_changes]
            if valid_genes:
                # Sort genes within group by average fold change
                avg_fold_changes = {gene: np.mean(fold_changes[gene]) for gene in valid_genes}
                sorted_group_genes = sorted(valid_genes, key=lambda g: avg_fold_changes[g], reverse=True)

                all_genes.extend(sorted_group_genes)
                current_position += len(sorted_group_genes)
                group_boundaries.append((group_name, current_position))

        # Create heatmap data array
        heatmap_data = np.array([fold_changes[gene] for gene in all_genes])

        # Create figure with adjusted height to accommodate groups
        plt.figure(figsize=(12, len(all_genes) * 0.4 + 1))

        # Create main heatmap
        sns.heatmap(heatmap_data,
                    annot=True,
                    fmt=".2f",
                    cmap="RdBu_r",
                    center=0,
                    xticklabels=pattern_names,
                    yticklabels=all_genes,
                    cbar_kws={'label': 'Log2 Fold Change (PA/NA)'})

        # Add significance asterisks
        for i, gene in enumerate(all_genes):
            for j in range(len(pattern_names)):
                if p_values[gene][j] < 0.001:
                    plt.text(j + 0.7, i + 0.5, '***', ha='center', va='center')
                elif p_values[gene][j] < 0.01:
                    plt.text(j + 0.7, i + 0.5, '**', ha='center', va='center')
                elif p_values[gene][j] < 0.05:
                    plt.text(j + 0.7, i + 0.5, '*', ha='center', va='center')

        # Add group labels and boundaries
        prev_pos = 0
        for group_name, end_pos in group_boundaries:
            middle_pos = prev_pos + (end_pos - prev_pos)/2
            plt.text(-0.5, middle_pos, group_name,
                    ha='right', va='center',
                    fontweight='bold')
            if end_pos < len(all_genes):
                plt.axhline(y=end_pos, color='white', linewidth=2)
            prev_pos = end_pos

        # Adjust layout and labels
        plt.title('Gene Expression Differences (PA vs NA) Across Expansion Patterns\n(Significant Genes Only)')
        plt.xlabel('Expansion Pattern')
        plt.ylabel('Genes')

        # Save plots with tight layout
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"gene_expression_heatmap_k{k}.pdf"), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(outdir, f"gene_expression_heatmap_k{k}.png"), format='png', dpi=300, bbox_inches='tight')
        plt.close()

        # Save numerical results
        results_df = pd.DataFrame({
            'Gene': all_genes,
            'Group': [next(group_name for group_name, end_pos in group_boundaries if end_pos > i)
                     for i in range(len(all_genes))],
            **{f"{pattern}_fold_change": [fold_changes[gene][i] for gene in all_genes]
               for i, pattern in enumerate(pattern_names)},
            **{f"{pattern}_pvalue": [p_values[gene][i] for gene in all_genes]
               for i, pattern in enumerate(pattern_names)}
        })
        results_df.to_csv(os.path.join(outdir, f"expression_analysis_results_k{k}.csv"), index=False)

        return results_df

    # Data preprocessing
    sc.pp.normalize_total(filtered_raw_adata, target_sum=1e4)
    sc.pp.log1p(filtered_raw_adata)
    os.makedirs(outdir, exist_ok=True)

    p_values = {gene: [] for gene in GENES_OF_INTEREST}  # dict of list
    fold_changes = {gene: [] for gene in GENES_OF_INTEREST}  # dict of list
    pattern_sizes = {}
    pattern_pa_sizes = {}
    # main loop: process each pattern
    for pattern, pattern_name in zip(SITE_PATTERNS, PATTERN_NAMES):
        # Filter data for current pattern
        pattern_adata = filtered_raw_adata[filtered_raw_adata.obs['pattern'].isin(pattern)].copy()

        # Create a temporary grouping column for DE analysis
        pattern_adata.obs['comparison_group'] = 'NA'
        pa_mask = pd.Series(False, index=pattern_adata.obs.index)
        for k in range(top_k):
            match_col = f'match_{k}'
            if match_col in pattern_adata.obs.columns:
                pa_mask |= (pattern_adata.obs[match_col] == 1)
        # Assign PA label to cells with any match in top K
        pattern_adata.obs.loc[pa_mask, 'comparison_group'] = 'PA'
        # Print statistics about the grouping
        n_pa = sum(pattern_adata.obs['comparison_group'] == 'PA')
        n_total = len(pattern_adata.obs)
        print(f"Num of PA cells (top {top_k} matches): {n_pa} ({n_pa/n_total*100:.2f}%) out of total {n_total} cells")

        pattern_sizes[pattern_name] = n_total
        pattern_pa_sizes[pattern_name] = n_pa

        # Perform differential gene expression analysis
        p_values, fold_changes = perform_deg_analysis(pattern_adata, GENES_OF_INTEREST, p_values, fold_changes)

    pattern_names_with_counts = [
        f"{pattern}\n(Total N={pattern_sizes[pattern]:,}\nPA N={pattern_pa_sizes[pattern]:,}, {pattern_pa_sizes[pattern]/pattern_sizes[pattern]*100:.1f}%)"
        for pattern in PATTERN_NAMES
    ]
    results_df = create_gene_expression_heatmap(fold_changes, p_values, GENE_GROUPS, pattern_names_with_counts, outdir, top_k)
    return results_df


def expression_level_analysis_grouped(filtered_raw_adata, outdir, top_k=1):
    def perform_deg_analysis(adata, genes_of_interest, top_k=1):
        """Perform DEG analysis between PA and NA cells"""
        # Initialize all cells as NA
        adata.obs['comparison_group'] = 'NA'

        # Create mask for PA cells considering top K matches
        pa_mask = pd.Series(False, index=adata.obs.index)
        for k in range(top_k):
            match_col = f'match_{k}'
            if match_col in adata.obs.columns:
                pa_mask |= (adata.obs[match_col] == 1)

        # Assign PA label to cells with any match in top K
        adata.obs.loc[pa_mask, 'comparison_group'] = 'PA'

        # Print statistics about the grouping
        n_pa = sum(adata.obs['comparison_group'] == 'PA')
        n_total = len(adata.obs)
        print(f"Num of PA cells (top {top_k} matches): {n_pa} ({n_pa/n_total*100:.2f}%) out of total {n_total} cells")

        # Perform DEG analysis
        sc.tl.rank_genes_groups(
            adata,
            groupby='comparison_group',
            groups=['PA'],  # Compare PA vs rest (NA)
            reference='NA',
            method='wilcoxon',
            key_added='deg_results',
            pts=True,  # Calculate percentage of cells expressing genes
            genes_batches=genes_of_interest  # Analyze only genes of interest
        )

        # Extract results
        results = {gene: {} for gene in genes_of_interest}
        for gene in genes_of_interest:
            if gene in adata.var_names:
                idx = [x[0] for x in adata.uns['deg_results']['names']].index(gene)
                results[gene] = {
                    'logfoldchange': [x[0] for x in adata.uns['deg_results']['logfoldchanges']][idx],
                    'pval': [x[0] for x in adata.uns['deg_results']['pvals']][idx],
                    'pval_adj': [x[0] for x in adata.uns['deg_results']['pvals_adj']][idx]
                }
            else:
                print(f"Warning: Gene {gene} not found in the dataset.")
        return results

    def create_grouped_gene_expression_plot(gene_results, outdir, gene_groups, top_k=1):
        """Create grouped visualization of DEG results"""
        plt.figure(figsize=(15, 10))
        plt.ylim(-2, 2)  # Set limits for log fold change scale

        # Calculate positions for grouped bars
        group_positions = []
        current_pos = 0

        for group_name in gene_groups.keys():
            group_genes = [g for g in gene_groups[group_name] if g in gene_results]
            if group_genes:
                group_positions.append((group_name, current_pos, current_pos + len(group_genes)))
                current_pos += len(group_genes) + 1.5

        def get_stars(pvalue):
            if pvalue <= 0.001:
                return '***'
            elif pvalue <= 0.01:
                return '**'
            elif pvalue <= 0.05:
                return '*'
            return ''

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
        plt.title("Differential Gene Expression (Log2 Fold Change PA vs NA)\nAll Cell Types",
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
        plt.savefig(os.path.join(outdir, f"deg_grouped_topk_{top_k}.pdf"),
                    format='pdf', bbox_inches='tight', dpi=300)
        print(os.path.join(outdir, f"deg_grouped_topk_{top_k}.pdf"))
        plt.close()

    # Preprocessing
    sc.pp.normalize_total(filtered_raw_adata, target_sum=1e4)
    sc.pp.log1p(filtered_raw_adata)
    os.makedirs(outdir, exist_ok=True)

    # Main analysis
    print("Performing differential expression analysis...")
    gene_results = perform_deg_analysis(filtered_raw_adata, GENES_OF_INTEREST, top_k=top_k)

    print("Creating visualization...")
    create_grouped_gene_expression_plot(gene_results, outdir, gene_groups=GENE_GROUPS, top_k=top_k)


def antigen_analysis2(mdata, outdir, top_k=1, color_criterion='cell_type'):
    assert color_criterion in ['cell_type', 'site_pattern']
    def draw_diagram(df_subset, outdir, limit_NAT=True, x_off=0.0, y_off=0.0, desc='none', show_clone_size=False, top_k=1, color_criterion='cell_type'):
        fig, ax = plt.subplots(figsize=(12, 10))

        # Draw the main rectangle
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False))

        if limit_NAT:
            # Draw the horizontal line
            ax.axhline(y=0.3, color='black', linestyle='--')
            # Fill NAT area with dull dark color
            ax.add_patch(plt.Rectangle((0, 0), 1, 0.3, facecolor='#4a4a4a', alpha=0.3))
            # Add labels
            ax.text(0.5, 0.95, 'Lung + Blood', ha='center', va='center', fontsize=12, fontweight='bold')
            ax.text(0.5, 0.15, 'NAT', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        else:
            ax.text(0.5, 0.95, 'Lung + NAT + Blood', ha='center', va='center', fontsize=12, fontweight='bold')

        # Define colors for cell types
        color_map = {'8.1-Teff': '#5A69AF', '8.2-Tem': '#579E65', '8.3a-Trm': '#F9C784', '8.3b-Trm': '#FC944A', '8.3c-Trm': '#F24C00'}

        # Prepare data for BubbleChart
        clone_sizes = df_subset['clone_size'].values
        if color_criterion == 'cell_type':
            # Define colors for cell types
            color_map = {'8.1-Teff': '#5A69AF', '8.2-Tem': '#579E65', '8.3a-Trm': '#F9C784', '8.3b-Trm': '#FC944A', '8.3c-Trm': '#F24C00'}
            cell_types = df_subset['ident'].values
        else:
            # Label pattern name
            def classify(pattern):
                if pattern in SITE_PATTERNS_CORE[1]:
                    return 'Tumor Singleton'
                elif pattern in SITE_PATTERNS_CORE[2]:
                    return 'Tumor Multiplet'
                elif pattern in SITE_PATTERNS_CORE[3]:
                    return 'Dual Expanded'
                else:
                    return 'Others'
            df_subset['pattern_name'] = df_subset['pattern'].apply(lambda x: classify(x))

            color_map = {'Tumor Singleton': '#5A69AF', 'Tumor Multiplet': '#579E65', 'Dual Expanded': '#F24C00', 'Others': '#F9C784'}
            cell_types = df_subset['pattern_name'].values

        colors = [color_map.get(cell_type, '#CCCCCC') for cell_type in cell_types]

        # Create and collapse BubbleChart
        bubble_chart = BubbleChart(area=clone_sizes, bubble_spacing=0.010)
        bubble_chart.collapse()

        # Scale and shift the bubbles to fit in the Lung + Blood area
        bubbles = bubble_chart.bubbles
        x_scale = 0.9 / (bubbles[:, 0].max() - bubbles[:, 0].min())
        y_scale = 0.6 / (bubbles[:, 1].max() - bubbles[:, 1].min())
        scale = min(x_scale, y_scale) * 0.8
        bubbles[:, :2] *= scale
        bubbles[:, 0] += x_off
        bubbles[:, 1] += y_off

        # Plot bubbles
        for i, (x, y, r, a) in enumerate(bubbles):
            circle = plt.Circle((x, y), r * scale, color=colors[i])
            ax.add_patch(circle)
            if show_clone_size:
                ax.text(x, y, str(int(a)), fontsize=8)
            # if df_subset.iloc[i]['match_0'] == 1:
            if df_subset.iloc[i][[f'match_{j}' for j in range(top_k)]].any():
                edge = plt.Circle((x, y), r * scale * 0.9, fill=False, edgecolor='yellow', linewidth=2)
                ax.add_patch(edge)

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Set title
        if limit_NAT:
            title_type = "Tumor Multiplet"
        else:
            title_type = "Dual Expansion"
        ax.set_title(f'Clonal Expansion Diagram for {title_type}', fontsize=14, fontweight='bold')

        # Add legend for cell types
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=cell_type,
                           markerfacecolor=color, markersize=10)
                           for cell_type, color in color_map.items()]
        ax.legend(handles=legend_elements, title='Cell Types', loc='center left', bbox_to_anchor=(1, 0.5))

        # Set aspect ratio to equal for circular bubbles
        ax.set_aspect('equal')

        # Save the figure
        plt.tight_layout()
        Path(f"{outdir}/{desc}").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{outdir}/{desc}/diagram_x_{x_off}_y_{y_off}_k{top_k}.pdf", dpi=300, bbox_inches='tight', format='pdf')
        plt.close()

    def draw_top_proteins_plot(df_subset, outdir, desc, top_k=1):
        # Create mask for rows with any positive match
        cols = [f'match_{j}' for j in range(top_k)]
        match_mask = df_subset[cols].any(axis=1)

        # Get all ref_protein columns for top_k matches
        ref_protein_cols = [f'ref_protein_{j}' for j in range(top_k)]

        # Filter rows with any match and sort by clone size
        matched_df = df_subset[match_mask].sort_values('clone_size', ascending=False)

        # Collect all proteins from matched rows (considering all ref_protein columns)
        all_proteins = []
        for _, row in matched_df.iterrows():
            # Get all non-NaN proteins for this row
            row_proteins = [str(row[col]) for col in ref_protein_cols if pd.notna(row[col])]
            all_proteins.extend(row_proteins)

        # Get top 5 unique proteins
        top_proteins = pd.Series(all_proteins).value_counts().head(5).index.tolist()

        # Create the plot
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.axis('off')  # Hide axes
        ax.text(0.5, 0.5, '\n'.join(top_proteins), va='center', ha='center', fontsize=10, fontweight='bold')

        # Save the figure as a PDF
        Path(f"{outdir}/{desc}").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{outdir}/{desc}/top_proteins_k{top_k}.pdf", dpi=300, bbox_inches='tight', format='pdf')
        plt.close()

    def process_dataframe(mdata):
        """Initial data processing common to all analyses."""
        df = mdata['gex'].obs
        df = get_cd8_tcells_with_tcrs(df)
        print(f"(antigen_analysis2) Consider only rows that are CD8+ T cells and have CDR3b sequences.")
        df['clone_size'] = df.groupby('cdr3_nt')['cdr3_nt'].transform('count')
        return df

    def get_filtered_clones(df, patterns, n_clones=100):
        """Filter dataframe based on patterns and return top N unique clones."""
        df_filtered = df[df['pattern'].isin(patterns)]
        df_sorted = df_filtered.sort_values('clone_size', ascending=False)
        df_unique = df_sorted.drop_duplicates(subset='cdr3_nt', keep='first')
        df_filtered = df_unique.head(n_clones)
        return df_filtered[df_filtered['clone_size'] != 0]

    def create_visualizations(df_filtered, outdir, desc, limit_NAT, show_clone_size=False, top_k=None, color_criterion='cell_type'):
        # Create all visualizations for a given filtered dataset.
        for x in [-0.4, -0.3, -0.1]:
            for y in [0.1, 0.3, 0.5]:
                draw_diagram(df_filtered, outdir,
                            limit_NAT=limit_NAT,
                            x_off=x, y_off=y,
                            desc=desc,
                            top_k=top_k,
                            color_criterion=color_criterion)
                if show_clone_size:
                    draw_diagram(df_filtered, outdir,
                               limit_NAT=limit_NAT,
                               x_off=x, y_off=y,
                               desc=f"{desc}_c",
                               show_clone_size=True,
                               top_k=top_k,
                                color_criterion=color_criterion)

    def analyze_patient_data(df, patient, patterns, outdir, analysis_type, top_k=None, color_criterion='cell_type'):
        """Analyze data for a specific patient and pattern type."""
        # Filter for specific patient
        df_patient = df[df['patient'] == patient]

        # Get filtered clones for this patient
        df_filtered = get_filtered_clones(df_patient, patterns)

        # Create visualizations with patient-specific descriptions
        create_visualizations(
            df_filtered,
            outdir,
            desc=f'{patient}_{analysis_type}',
            limit_NAT=(analysis_type == 'TM'),
            show_clone_size=True,
            top_k=top_k,
            color_criterion=color_criterion
        )

    # Create output directory
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Process initial dataframe
    df = process_dataframe(mdata)

    # Define patterns
    tumor_multiplet_patterns = ['Txb', 'TxB', 'Txx']
    DE_patterns = ['tnb', 'tnB', 'tNb', 'tNB', 'Tnb', 'TnB', 'TNb', 'TNB',
                  'tNx', 'tnx', 'Tnx', 'TNx']

    # (1) Analyze Tumor Multiplet T cells
    # df_tm_filtered = get_filtered_clones(df, tumor_multiplet_patterns)
    # create_visualizations(df_tm_filtered, outdir, 'TM',
    #                     limit_NAT=True, show_clone_size=True, top_k=top_k)

    # # (2) Analyze Dual Expanded T cells
    # df_de_filtered = get_filtered_clones(df, DE_patterns)
    # create_visualizations(df_de_filtered, outdir, 'DE',
    #                     limit_NAT=False, show_clone_size=False, top_k=top_k)

    # Get unique patients
    patients = df['patient'].unique()

    for patient in patients:
        if color_criterion == 'cell_type':
            # (3) Analyze Tumor Multiplet patterns (per-patient)
            analyze_patient_data(df, patient, tumor_multiplet_patterns,
                                 outdir, 'TM', top_k, color_criterion)

            # (4) Analyze Dual Expanded patterns (per-patient)
            analyze_patient_data(df, patient, DE_patterns,
                                 outdir, 'DE', top_k, color_criterion)
        else:  # based on site_pattern
            analyze_patient_data(df, patient, SITE_PATTERNS_CORE[0],
                                 outdir, 'All', top_k, color_criterion)
