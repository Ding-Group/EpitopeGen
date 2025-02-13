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
    """Class for performing Phenotype-Association Ratio analysis."""

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
            cell_types: List of cell types to analyze
            pattern_names: List of pattern names
            pattern_descriptions: Dictionary mapping pattern names to descriptions
            patterns_dict: List of lists containing site patterns to analyze
            output_dir: Directory to save results (default: "analysis/PA_ratio_analysis")
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
        """Validate input parameters."""
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

    def analyze(
        self,
        mdata: Dict,
        top_k: int = 8,
        per_patient: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Perform PA ratio analysis.

        Args:
            mdata: Dictionary containing gene expression data
            top_k: Number of top matches to consider
            per_patient: Whether to perform analysis per patient

        Returns:
            Dictionary containing analysis results DataFrames
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
        """Preprocess the input data."""
        # Implement your preprocessing logic here
        # This is a placeholder - replace with actual implementation
        return df

    def _calculate_statistics(
        self,
        df: pd.DataFrame,
        n_gen: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Calculate PA ratios and statistics."""
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
        """Apply multiple testing correction to p-values."""
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
        """Create visualizations for a pattern group."""
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
        pattern_names = data.index.tolist()

        """Create bar plot of PA ratios with significance markers."""
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
        """Set up the bar plot figure and axes."""
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
        """Plot bars and add significance markers."""
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
        """Add significance markers to bars."""
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
        """Customize bar plot appearance."""
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
        max_ratio = np.ceil(melted_data['PA Ratio'].max() * 10) / 10
        ax.set_ylim(0, max_ratio * 1.2)

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
        """Save analysis results to CSV files.

        Args:
            ratio_df: DataFrame containing PA ratios
            p_value_df: DataFrame containing corrected p-values
            num_df: DataFrame containing cell counts
            n_gen: Number of generations used for matching
        """
        ratio_df.to_csv(self.output_dir / f'PA_ratio_all_patients_k{n_gen}.csv')
        p_value_df.to_csv(self.output_dir / f'p_value_all_patients_k{n_gen}.csv')
        num_df.to_csv(self.output_dir / f'num_cells_all_patients_k{n_gen}.csv')


if __name__ == "__main__":
    from itertools import combinations, combinations_with_replacement, product
    first_char = ['T', 't', 'x']
    second_char = ['N', 'n', 'x']
    third_char = ['B', 'b', 'x']

    analyzer = PARatioAnalyzer(
        cell_types=['8.1-Teff', '8.2-Tem', '8.3a-Trm', '8.3b-Trm', '8.3c-Trm'],
        pattern_names=['All', 'Tumor singleton', 'Tumor multiplet', 'DE'],
        pattern_descriptions={
            'All': 'All',
            'Tumor singleton': 'Tumor Singleton (TS)',
            'Tumor multiplet': 'Tumor Multiplet (TM)',
            'DE': 'Dual Expansion (DE)'
        },
        patterns_dict={
            'All': [''.join(combo) for combo in product(first_char, second_char, third_char)],  # [1] all
            'Tumor singleton': ['txb', 'txB', 'txx'],  # [5] tumor singleton
            'Tumor multiplet': ['Txb', 'TxB', 'Txx'],  # [6] tumor multiplet
            'DE': ['tnb',  'tnB',  'tNb', 'tNB', 'Tnb', 'TnB', 'TNb', 'TNB', 'tNx', 'tnx', 'Tnx', 'TNx']  # [7] Dual expansion
        },
        output_dir="example_run/PA_ratio"
    )

    cfg = {
        "exp_dir": "example_run",
        "input_file": "data/sample_tcrs.csv",
        "epitope_db": "data/tumor_associated_epitopes.csv",
        "top_k": 4,
        "method": 'substring',
        "ens_th": 0.5,
    }

    # from research.cancer_wu.utils import *
    # from research.cancer_wu.analyze import *
    import scanpy as sc
    import pandas as pd
    import os
    import anndata as ad
    import scirpy as ir
    import mudata as md
    from mudata import MuData

    CELL_TYPES = ['8.1-Teff', '8.2-Tem', '8.3a-Trm', '8.3b-Trm', '8.3c-Trm']
    SAMPLES = ['CN1', 'CN2', 'CT1', 'CT2', 'EN1', 'EN2', 'EN3', 'ET1', 'ET2', 'ET3',
               'LB6', 'LN1', 'LN2', 'LN3', 'LN4', 'LN5', 'LN6', 'LT1', 'LT2', 'LT3',
               'LT4', 'LT5', 'LT6', 'RB1', 'RB2', 'RB3', 'RN1', 'RN2', 'RN3', 'RT1',
               'RT2', 'RT3']

    def read_tcell_integrated(data_dir, transpose=False):
        """
        Read the main gene expression data
        """
        # Read the H5AD file
        adata = sc.read_h5ad(f"{data_dir}/GSE139555_tcell_integrated.h5ad")
        if transpose:
            adata = adata.transpose()
        metadata = pd.read_csv(f"{data_dir}/GSE139555%5Ftcell%5Fmetadata.txt", sep="\t", index_col=0)
        # Make sure the index of the metadata matches the obs_names of the AnnData object
        adata.obs = adata.obs.join(metadata, how='left')
        print("Successfully read GSE139555_t_cell_integrated!")
        return adata


    def read_all_data(data_dir, obs_cache=None, filter_cdr3_notna=True, filter_cell_types=True):
        """
        The main function to read CD8+ T cell data from Wu et al. dataset
        Both gene expression and TCR sequences are read

        Parameters
        ----------
        data_dir: str
            Root directory of the data
        obs_cache: str / None
            csv file that contains some annotated TCR data. As there are multiple annotation steps,
            this file is always read after the very first annotation
        filter_cdr3_notna: bool
            Drop the rows that do not have viable CDR3 sequence information
        filter_cell_types: bool
            Drop the rows that are not CD8+ T cells
        """
        samples = ['CN1', 'CT2', 'EN3', 'ET3', 'LB6', 'LN3', 'LN6', 'LT3', 'LT6', 'RB2', 'RN2', 'RT2',
                   'CN2', 'EN1', 'ET1', 'LN1', 'LN4', 'LT1', 'LT4', 'RB3', 'RN3', 'RT3',
                   'CT1', 'EN2', 'ET2', 'LN2', 'LN5', 'LT2', 'LT5', 'RB1', 'RN1', 'RT1']
        # Read T-cell integrated (gene expression data)
        adata = read_tcell_integrated(data_dir)

        # Read the TCR sequencing data using scirpy (ir)
        airrs = []
        for sample in [s for s in os.listdir(data_dir) if s in samples]:
            for x in os.listdir(f"{data_dir}/{sample}"):
                if x.endswith("contig_annotations.csv") or x.endswith("annotations.csv"):
                    airr = ir.io.read_10x_vdj(f"{data_dir}/{sample}/{x}")
                    # Add a column to identify the source file
                    airr.obs['new_cell_id'] = airr.obs.index.map(lambda x: sample + "_" + x)
                    airr.obs.index = airr.obs['new_cell_id']
                    airrs.append(airr)
        # Merge the AIRR objects
        if len(airrs) > 1:
            merged_airr = ad.concat(airrs)
        else:
            merged_airr = airrs[0]

        if obs_cache:
            print(f"Reading cache from {obs_cache}..")
            df_cache = pd.read_csv(obs_cache)

            # Merge df_cache to adata.obs based on cell_id
            # Set cell_id as index in df_cache to match adata.obs
            df_cache = df_cache.set_index('cell_id')

            # Keep only the cells that exist in df_cache
            common_cells = adata.obs.index.intersection(df_cache.index)
            adata = adata[common_cells].copy()

            # Update adata.obs with all columns from df_cache
            # This will overwrite existing columns and add new ones
            adata.obs = adata.obs.combine_first(df_cache)

            # For columns that exist in both, prefer df_cache values
            for col in df_cache.columns:
                if col in adata.obs:
                    adata.obs[col] = df_cache[col]

            print(f"Updated adata.obs with {len(df_cache.columns)} columns from cache")
            print(f"Retained {len(common_cells)} cells after matching with cache")

        if filter_cell_types:
            print("Get only CD8+ T cells..")
            adata = adata[adata.obs['ident'].isin(CELL_TYPES)].copy()

        if filter_cdr3_notna:
            # Filter based on non-NA cdr3 values:
            valid_cells = adata.obs['cdr3'].notna()
            print(f"Filtering out {(~valid_cells).sum()} cells with NA cdr3 values")
            adata = adata[valid_cells].copy()

        mdata = MuData({"airr": merged_airr, "gex": adata})

        print(f"Successfully merged {len(airrs)} AIRR objects!")
        print(f"(read_all_data) The number of CD8+ T cells: {len(adata.obs)}")
        return mdata


    def read_all_raw_data(data_dir):
        samples = os.listdir(data_dir)
        adata_list = []

        for sample in samples:
            if sample in SAMPLES:
                sample_path = os.path.join(data_dir, sample)
                file = os.listdir(sample_path)[0]

                # Read the data with the prefix applied to barcodes
                adata = sc.read_10x_mtx(
                    path=sample_path,
                    var_names="gene_symbols",
                    make_unique=True,
                    prefix=file.split(".")[0] + "."
                )

                # Rename the barcodes
                prefix = f"{sample}_"
                adata.obs_names = [f"{prefix}{barcode}" for barcode in adata.obs_names]

                # Append the annotated data to the list
                adata_list.append(adata)

        # Concatenate all the data into one AnnData object
        combined_adata = ad.concat(adata_list, axis=0)
        print("Successfully read all RAW data!")

        return combined_adata


    def filter_and_update_combined_adata(combined_adata, processed_adata):
        # Get the common indices (barcodes) between the combined_adata and processed_adata
        common_indices = processed_adata.obs_names.intersection(combined_adata.obs_names)

        # Filter combined_adata to keep only those cells present in processed_adata
        filtered_combined_adata = combined_adata[common_indices].copy()

        # Copy obs from processed_adata to filtered_combined_adata
        for col in processed_adata.obs.columns:
            # Add a new column in filtered_combined_adata if it doesn't already exist
            if col not in filtered_combined_adata.obs.columns:
                filtered_combined_adata.obs[col] = None

            # Copy the data from processed_adata.obs to filtered_combined_adata.obs, matching by index
            filtered_combined_adata.obs[col] = processed_adata.obs.loc[common_indices, col]

        print(f"Filtered the combined data using the processed adata! (Finding intersection). Num of rows={len(filtered_combined_adata)}")

        return filtered_combined_adata

    from epigen import DEGAnalyzer
    mdata = read_all_data(data_dir="data/cancer_wu", obs_cache=f"{cfg['exp_dir']}/obs_annotated_cancer_wu_ens_th0.5_merged_annotations_ens_all_th0.5.csv")
    analyzer.analyze(mdata, top_k=2)
