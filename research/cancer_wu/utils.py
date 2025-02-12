import os
from collections import defaultdict
from functools import partial
from itertools import combinations, combinations_with_replacement, product
from multiprocessing import Pool, cpu_count
from pathlib import Path

# Third-party imports
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scirpy as ir
import seaborn as sns
from mudata import MuData
from tqdm import tqdm


CELL_TYPES = ['8.1-Teff', '8.2-Tem', '8.3a-Trm', '8.3b-Trm', '8.3c-Trm']
first_char = ['T', 't', 'x']
second_char = ['N', 'n', 'x']
third_char = ['B', 'b', 'x']
SITE_PATTERNS = [
    [''.join(combo) for combo in product(first_char, second_char, third_char)],  # [1] all
    ['TNx', 'tNx', 'Tnx', 'tnx'],  # [2] Dual expansion, Blood independent
    ['TNb', 'tNb', 'tnb', 'Tnb'],  # [3] Dual expansion, Blood non-expanded
    ['TNB', 'tNB', 'tnB', 'TnB'],  # [4] Dual expansion, Blood expanded
    ['txb', 'txB', 'txx'],  # [5] tumor singleton
    ['Txb', 'TxB', 'Txx'],  # [6] tumor multiplet
    ['tnb',  'tnB',  'tNb', 'tNB', 'Tnb', 'TnB', 'TNb', 'TNB', 'tNx', 'tnx', 'Tnx', 'TNx']  # [7] Dual expansion
]
PATTERN_NAMES = [
    'All',
    'DE, Blood Independent',
    'DE, Blood Non-expanded',
    'DE, Blood-expanded',
    'Tumor singleton',
    'Tumor multiplet',
    'DE'
]
PATTERN_NAMES2DESC = {
    'All': 'All',
    'DE, Blood Independent': 'Dual Expansion, Blood Independent (DEBI)',
    'DE, Blood Non-expanded': 'Dual Expansion, Blood Non-expanded (DEBN)',
    'DE, Blood-expanded': 'Dual Expansion, Blood Expanded (DEBE)',
    'Tumor singleton': 'Tumor Singleton (TS)',
    'Tumor multiplet': 'Tumor Multiplet (TM)',
    'DE': 'Dual Expansion (DE)'
}
SITE_PATTERNS_CORE = [
    [''.join(combo) for combo in product(first_char, second_char, third_char)],  # [1] all
    ['txb', 'txB', 'txx'],  # [2] tumor singleton
    ['Txb', 'TxB', 'Txx'],  # [3] tumor multiplet
    ['tnb', 'tnB', 'tNb', 'tNB', 'Tnb', 'TnB', 'TNb', 'TNB', 'tNx', 'tnx', 'Tnx', 'TNx']  # [4] Dual expansion
]
PATTERN_NAMES_CORE = ['All', 'Tumor singleton', 'Tumor multiplet', 'DE']
GENES_OF_INTEREST = [
    'PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'CD244',  # exhaustion markers
    'CD69', 'IL2RA', 'MKI67', 'GZMB', 'PRF1',  # activation and proliferation markers
    'ZAP70', 'LAT', 'LCK',  # TCR pathway that can indicate antigen recognition
    'IFNG', 'TNF',  # cytokines
    'TOX', 'EOMES', 'TBX21',  # exhaustion-assoc TFs
    'SLC2A1', 'PRKAA1',  # metabolic markers
    'HIF1A', 'XBP1',  # stress response
    'CCR7', 'IL7R',  # naive T cells (bystanders)
    'KLRG1', 'CX3CR1',  # terminally differentiated
    'ENTPD1'  # CD39: recently found marker
]
GENE_GROUPS = {
    'Cytotoxicity': ['GZMB', 'PRF1'],
    'Naive/Memory markers': ['CCR7', 'IL7R'],
    'Early activation': ['CD69', 'IL2RA'],
    'Proliferation': ['MKI67'],
    'Exhaustion markers': ['PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'CD244', 'ENTPD1'],
    'Terminal differentiation': ['KLRG1', 'CX3CR1'],
    'TCR signaling': ['ZAP70', 'LAT', 'LCK'],
    'Cytokines': ['IFNG', 'TNF'],
    'Transcription factors': ['TOX', 'EOMES', 'TBX21'],
    'Metabolic regulators': ['SLC2A1', 'PRKAA1'],
    'Stress response': ['HIF1A', 'XBP1']
}
SAMPLES = ['CN1', 'CN2', 'CT1', 'CT2', 'EN1', 'EN2', 'EN3', 'ET1', 'ET2', 'ET3',
           'LB6', 'LN1', 'LN2', 'LN3', 'LN4', 'LN5', 'LN6', 'LT1', 'LT2', 'LT3',
           'LT4', 'LT5', 'LT6', 'RB1', 'RB2', 'RB3', 'RN1', 'RN2', 'RN3', 'RT1',
           'RT2', 'RT3']


def get_cd8_tcells_with_tcrs(df):
    """
    Get only rows that are CD8+ T cells and have viable TCR sequences
    """
    df = df[df['cdr3'].notna()]
    df = df[df['ident'].isin(CELL_TYPES)]
    print(f"The data subset of interest contains total {len(df)} CD8+ T cells. ")
    return df


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


class BubbleChart:
    def __init__(self, area, bubble_spacing=0):
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3])

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0], bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        idx_min = np.argmin(distance)
        return idx_min if type(idx_min) == np.ndarray else np.array([idx_min])

    def collapse(self, n_iterations=50):
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                dir_vec = self.com - self.bubbles[i, :2]
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        new_point1 = self.bubbles[i, :2] + orth * self.step_dist
                        new_point2 = self.bubbles[i, :2] - orth * self.step_dist
                        dist1 = self.center_distance(self.com, np.array([new_point1]))
                        dist2 = self.center_distance(self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()
            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2


def compare_predictions(file1_path, file2_path, top_k=1, verbose=False):
    """
    Compare predictions between two files considering top K predictions for each TCR.
    A match is found if any prediction in top K from file1 matches any prediction in top K from file2.

    Args:
        file1_path (str): Path to first CSV file
        file2_path (str): Path to second CSV file
        top_k (int): Number of top predictions to consider (default: 1)
        verbose (bool): Whether to print detailed results (default: False)
    """
    # Read both CSV files
    df1 = pd.read_csv(file1_path, skipinitialspace=True)
    df2 = pd.read_csv(file2_path, skipinitialspace=True)

    # Set TCR as index for easier matching
    df1.set_index('tcr', inplace=True)
    df2.set_index('tcr', inplace=True)

    # Get common TCRs
    common_tcrs = set(df1.index) & set(df2.index)
    total_tcrs = len(common_tcrs)

    if total_tcrs == 0:
        print("No common TCRs found between the files")
        return

    # Counter for matching predictions
    matching_count = 0

    # Compare predictions for each common TCR
    for tcr in tqdm(common_tcrs):
        # Get top K predictions from both files
        preds1 = [df1.loc[tcr, f'pred_{i}'][:9] for i in range(top_k)]
        preds2 = [df2.loc[tcr, f'pred_{i}'][:9] for i in range(top_k)]

        # Check if any prediction matches between the two sets
        has_match = any(pred1 == pred2
                       for pred1 in preds1
                       for pred2 in preds2)

        if has_match:
            matching_count += 1

    # Calculate similarity metrics
    similarity_percentage = (matching_count / total_tcrs) * 100

    result = {
        'file1_path': file1_path,
        'file2_path': file2_path,
        'sim': similarity_percentage
    }

    if verbose:
        # Print results
        print(f"Total common TCRs: {total_tcrs}")
        print(f"Number of matching predictions (top {top_k}): {matching_count}")
        print(f"Between {file1_path} & {file2_path}")
        print(f"(k={top_k}) Similarity percentage: {similarity_percentage:.2f}%")

    return result


def analyze_match_overlap(file1_path, file2_path, top_k=1):
    """
    Analyze overlap between matched TCRs in two files considering top K matches.
    A TCR is considered matched if any of its match_0 to match_{K-1} equals 1.

    Args:
        file1_path (str): Path to first CSV file
        file2_path (str): Path to second CSV file
        top_k (int): Number of top matches to consider (default: 1)
    """
    # Read both CSV files and drop NA values in cdr3
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    df1 = df1.dropna(subset=['cdr3'])
    df2 = df2.dropna(subset=['cdr3'])

    # Create masks for TCRs that have any match in top K predictions
    df1_match_mask = pd.Series(False, index=df1.index)
    df2_match_mask = pd.Series(False, index=df2.index)

    # Check each match column up to top_k
    for k in range(top_k):
        match_col = f'match_{k}'
        if match_col in df1.columns and match_col in df2.columns:
            df1_match_mask |= (df1[match_col] == 1)
            df2_match_mask |= (df2[match_col] == 1)

    # Get TCRs where any match in top K equals 1
    df1_matched = df1[df1_match_mask]
    df2_matched = df2[df2_match_mask]

    # Get unique TCRs for set operations
    matched_tcrs_1 = set(df1_matched['cdr3'])
    matched_tcrs_2 = set(df2_matched['cdr3'])

    # Calculate overlap
    overlap = matched_tcrs_1 & matched_tcrs_2

    # Calculate expected random overlap percentage
    total_tcrs = 89042  # Total population size
    size1 = len(matched_tcrs_1)  # Size of first matched set
    size2 = len(matched_tcrs_2)  # Size of second matched set

    # Expected overlap probability when drawing with replacement
    # expected_overlap_pct = (size1 / total_tcrs) * (size2 / total_tcrs) * 100
    expected_matches = size1 * (size2 / total_tcrs)
    expected_overlap_pct = expected_matches / min(size1, size2) * 100

    # Print statistics
    print(f"\n*** Between {file1_path} & {file2_path} (considering top {top_k} matches)")
    print(f"- Total TCRs: {len(df1)}")
    print(f"top_k = {top_k}")
    print(f"- Expected random overlap: {expected_overlap_pct:.2f}%")
    print(f"- File 1 matched TCRs: {len(matched_tcrs_1)}")
    print(f"- File 2 matched TCRs: {len(matched_tcrs_2)}")

    # Calculate percentage of overlap using dataframe counts
    if len(df1_matched) > 0:
        overlap_pct_1 = (len(df1_matched[df1_matched['cdr3'].isin(overlap)]) / len(df1_matched)) * 100
        print(f"- Percentage of File 1 matches that overlap: {overlap_pct_1:.2f}%")
    else:
        print("- No matches in File 1")

    if len(df2_matched) > 0:
        overlap_pct_2 = (len(df2_matched[df2_matched['cdr3'].isin(overlap)]) / len(df2_matched)) * 100
        print(f"- Percentage of File 2 matches that overlap: {overlap_pct_2:.2f}%")
    else:
        print("- No matches in File 2")

    # Return results as a dictionary for potential further analysis
    return {
        'file1_path': file1_path,
        'file2_path': file2_path,
        'total_tcrs': len(df1),
        'file1_matches': len(matched_tcrs_1),
        'file2_matches': len(matched_tcrs_2),
        'overlap_size': len(overlap),
        'overlap_pct_file1': overlap_pct_1 if len(df1_matched) > 0 else 0,
        'overlap_pct_file2': overlap_pct_2 if len(df2_matched) > 0 else 0
    }


def process_pair(pair, root):
    """
    Process a single pair of folders to compute similarity.

    Args:
        pair (tuple): Tuple containing indices and folder names (i, j, folder1, folder2)
        root (str): Root directory path

    Returns:
        tuple: (i, j, similarity_score)
    """
    i, j, folder1, folder2 = pair
    file1 = f"{root}/{folder1}/240826_wu_formatted.csv"
    file2 = f"{root}/{folder2}/240826_wu_formatted.csv"

    result = compare_predictions(file1, file2)
    return (i, j, result['sim'])

def extract_epoch_number(folder_name):
    """Extract epoch number from folder name"""
    try:
        # Extract number after 'e' in folder name
        epoch = int(folder_name.split('e')[-1])
        return epoch
    except:
        return float('inf')  # Handle cases where epoch number can't be extracted


def get_most_frequent(series, default=None):
    """
    Get the most frequent non-NaN value in a series.
    If all values are NaN or series is empty, returns default value.

    Args:
        series (pd.Series): Input series
        default: Value to return if no valid value found
    """
    # Remove NaN values
    valid_values = series.dropna()
    if len(valid_values) == 0:
        return default

    # Get value counts and return the most frequent
    return valid_values.mode().iloc[0]


def ensemble_PA_marked(PA_marked_list, outdir, th=0.5, desc='PA', col='match', num_cols=32):
    """
    Ensemble multiple PA_marked.csv files using majority voting based on cumulative matches
    at specific k-mer cutoffs (1, 2, 4, 8, 16, 32).
    Args:
        PA_marked_list (list): List of paths to PA_marked csv files
        outdir (str): Output directory path
        th (float): Threshold for majority voting (default: 0.5)
    """
    cutoff_indices = list(range(num_cols))

    Path(outdir).mkdir(parents=True, exist_ok=True)

    if col != 'match':  # include broader corona virus
        cols = ['match', col]
    else:
        cols = ['match']

    # Read all files and create cumulative matches for each cutoff
    dfs_with_cumul = []
    for data_path in PA_marked_list:
        df = pd.read_csv(data_path)

        # Create cumulative match columns for each cutoff
        for c in cols:
            for cutoff in cutoff_indices:
                match_cols = [f'{c}_{i}' for i in range(cutoff + 1)]
                cumul_col_name = f'cumulative_{c}_{cutoff}'
                df[cumul_col_name] = df[match_cols].any(axis=1).astype(int)

        dfs_with_cumul.append(df)

    # Initialize result dataframe with the first dataframe
    result_df = dfs_with_cumul[0].copy()

    # Perform majority voting for each cutoff
    for c in cols:
        for cutoff in cutoff_indices:
            cumul_col_name = f'cumulative_{c}_{cutoff}'

            # Stack predictions for this cutoff
            stacked_predictions = pd.concat(
                [df[cumul_col_name] for df in dfs_with_cumul],
                axis=1
            )

            # Perform majority voting
            majority_vote = (stacked_predictions.mean(axis=1) >= th).astype(int)

            # Update match column
            result_df[f"{c}_{cutoff}"] = majority_vote

            # Set the protein and epitope columns based on majority vote
            if c == 'match':
                prot_col_name = f'ref_protein_{cutoff}'
                epi_col_name = f'ref_epitope_{cutoff}'
            else:
                prot_col_name = f'corona_protein_{cutoff}'
                epi_col_name = f'corona_epitope_{cutoff}'

            # Stack protein predictions and get most frequent
            stacked_preds_prot = pd.concat([df[prot_col_name] for df in dfs_with_cumul], axis=1)
            result_df[prot_col_name] = stacked_preds_prot.apply(get_most_frequent, axis=1)

            # Stack epitope predictions and get most frequent
            stacked_preds_epi = pd.concat([df[epi_col_name] for df in dfs_with_cumul], axis=1)
            result_df[epi_col_name] = stacked_preds_epi.apply(get_most_frequent, axis=1)

            # For rows where match is 0, set protein and epitope to NaN
            mask = result_df[f"{c}_{cutoff}"] == 0
            result_df.loc[mask, prot_col_name] = np.nan
            result_df.loc[mask, epi_col_name] = np.nan

        # Remove temporary cumulative columns
        cumul_cols = [f'cumulative_{c}_{cutoff}' for cutoff in cutoff_indices]
        result_df = result_df.drop(columns=cumul_cols)

    # Save the ensembled results
    output_path = os.path.join(outdir, f'{desc}_marked_ensembled_th{th}.csv')
    result_df.to_csv(output_path, index=False)
    print(f"Saved ensembled results to: {output_path}")

    # Print statistics for each cutoff
    print("\nEnsemble Statistics:")
    print(f"Number of files ensembled: {len(PA_marked_list)}")
    for cutoff in cutoff_indices:
        match_cols = [f'{col}_{i}' for i in range(cutoff + 1)]
        is_positive = result_df[match_cols].any(axis=1)
        n_positive = is_positive.sum()
        n_total = len(result_df)
        k = cutoff + 1
        print(f"k={k}: Positive predictions: {n_positive} ({n_positive/n_total*100:.2f}%)")

    return result_df


def inspect_num_PA(PA_marked, col='match'):
    df = pd.read_csv(PA_marked)
    df = df[df['cdr3'].notna()]
    cell_types = ['8.1-Teff', '8.2-Tem', '8.3a-Trm', '8.3b-Trm', '8.3c-Trm']
    df = df[df['ident'].isin(cell_types)]
    print(f"Total {len(df)} cells.")
    # df = df[df['chain_pairing'] == 'Single pair']
    for k in [1,2,3,4,5,6,7,8]:
        match_cols = [f'{col}_{i}' for i in range(k)]
        N = len(df[df[match_cols].ge(1).any(axis=1)])
        print(f"{PA_marked} - k: {k}, N: {N}")


def analyze_pair(args, top_k=1):
    """
    Helper function to analyze a single pair of files.

    Args:
        args (tuple): Tuple containing (file1_path, file2_path)
        top_k (int): Number of top matches to consider
    """
    try:
        file1_path, file2_path = args
        result = analyze_match_overlap(file1_path, file2_path, top_k=top_k)
        overlap_value = (result['overlap_pct_file1'] + result['overlap_pct_file2']) / 2
        return (Path(file1_path).parts[2],
                Path(file2_path).parts[2],
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
    file_names = [Path(f).parts[2] for f in files_list]

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

    return overlap_matrix


def site_pattern_codes_hist(df_path, outdir, first_char, second_char, third_char):
    # Define SITE_PATTERNS_CORE and groups
    SITE_PATTERNS_CORE = [
        [''.join(combo) for combo in product(first_char, second_char, third_char)],  # [1] all
        ['txb', 'txB', 'txx'],  # [2] tumor singleton
        ['Txb', 'TxB', 'Txx'],  # [3] tumor multiplet
        ['tnb', 'tnB', 'tNb', 'tNB', 'Tnb', 'TnB', 'TNb', 'TNB', 'tNx', 'tnx', 'Tnx', 'TNx']  # [4] Dual expansion
    ]
    groups = {
        'Tumor Singleton': SITE_PATTERNS_CORE[1],
        'Tumor Multiplet': SITE_PATTERNS_CORE[2],
        'Dual Expanded': SITE_PATTERNS_CORE[3]
    }

    # Read the DataFrame
    df = pd.read_csv(df_path)

    # Filter the DataFrame using the provided function
    df = get_cd8_tcells_with_tcrs(df)
    print(f"Total of {len(df)} entries.")

    # Generate all possible codes
    all_codes = [''.join(combo) for combo in product(first_char, second_char, third_char)]

    # Calculate the frequency of each code
    code_frequencies = {code: len(df[df['pattern'] == code]) for code in all_codes}

    # Group the codes and sort within each group
    grouped_data = []
    for group_name, codes_in_group in groups.items():
        group_freq = {code: code_frequencies.get(code, 0) for code in codes_in_group}
        sorted_group = sorted(group_freq.items(), key=lambda x: x[1], reverse=True)
        grouped_data.append((group_name, sorted_group))

    # Add the remaining codes as a separate group
    grouped_codes = set(code for group in groups.values() for code in group)
    remaining_codes = [(code, code_frequencies.get(code, 0)) for code in all_codes if code not in grouped_codes]
    remaining_codes_sorted = sorted(remaining_codes, key=lambda x: x[1], reverse=True)
    grouped_data.append(('Other', remaining_codes_sorted))

    # Flatten the grouped data for plotting
    x_labels = []
    frequencies = []
    group_labels = []
    for group_name, sorted_group in grouped_data:
        for code, freq in sorted_group:
            x_labels.append(code)
            frequencies.append(freq)
            group_labels.append(group_name)

    # Plot the histogram
    plt.figure(figsize=(14, 6))
    colors = {
        'Tumor Singleton': 'skyblue',
        'Tumor Multiplet': 'lightgreen',
        'Dual Expanded': 'salmon',
        'Other': 'lightgray'
    }
    bar_colors = [colors[group] for group in group_labels]

    bars = plt.bar(x_labels, frequencies, color=bar_colors)
    plt.xlabel('Site Pattern Code')
    plt.ylabel('Frequency')
    plt.title('Frequency of Site Pattern Codes (Grouped)')
    plt.xticks(rotation=90)

    # Add legend for groups
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[group], label=group) for group in colors]
    plt.legend(handles=legend_elements, title="Groups")

    plt.tight_layout()

    # Save the plot as a PDF
    output_path = f"{outdir}/site_pattern_codes_histogram_grouped.pdf"
    plt.savefig(output_path)
    plt.close()

    print(f"Histogram saved to {output_path}")


def check_samples_numbers(df_path, hierarchy='celltype', check_PA=False, K=1):
    """
    Calculate and print the number of samples per cell type and pattern name.

    Parameters:
        df_path (str): Path to the DataFrame CSV file.
        hierarchy (str): Determines the printing hierarchy.
                         Options: 'pattern' (default) or 'celltype'.
    """
    # Read the DataFrame
    df = pd.read_csv(df_path)

    # Filter the DataFrame using the provided function
    df = get_cd8_tcells_with_tcrs(df)
    print(f"Total of {len(df)} entries.")

    # Define pattern names
    pattern_names = ['All', 'Tumor Singleton', 'Tumor Multiplet', 'Dual Expanded']

    # Initialize a nested dictionary to store the counts
    num_samples = {pattern_name: {cell_type: 0 for cell_type in CELL_TYPES} for pattern_name in pattern_names}

    # Calculate the number of samples for each cell type and pattern name
    if check_PA:
        match_columns = [f'match_{i}' for i in range(K)]

    for celltype in CELL_TYPES:
        for pattern_name, codes in zip(pattern_names, SITE_PATTERNS_CORE):
            if check_PA:
                num = len(df[(df['pattern'].isin(codes)) & (df['ident'] == celltype) & (df[match_columns].any(axis=1))])
            else:
                num = len(df[(df['pattern'].isin(codes)) & (df['ident'] == celltype)])
            num_samples[pattern_name][celltype] = num

    # Print the results based on the specified hierarchy
    if check_PA:
        print("We focus on PA T cells, because check_PA=True")

    if hierarchy == 'pattern':
        for pattern_name in pattern_names:
            print(f"\nPattern: {pattern_name}")
            for celltype in CELL_TYPES:
                print(f"  Cell Type: {celltype}, Number of Samples: {num_samples[pattern_name][celltype]}")
    elif hierarchy == 'celltype':
        for celltype in CELL_TYPES:
            print(f"\nCell Type: {celltype}")
            for pattern_name in pattern_names:
                print(f"  Pattern: {pattern_name}, Number of Samples: {num_samples[pattern_name][celltype]}")
    else:
        raise ValueError("Invalid hierarchy argument. Use 'pattern' or 'celltype'.")
