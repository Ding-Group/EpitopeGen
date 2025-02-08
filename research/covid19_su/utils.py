# Standard library imports
import os
from collections import defaultdict
from functools import partial
from itertools import combinations, combinations_with_replacement
from multiprocessing import Pool, cpu_count
from pathlib import Path

# Third-party imports
import csv
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from tqdm import tqdm


amino_acids = set('ACDEFGHIKLMNPQRSTVWY')

SIGNATURE_GENES = {
    'naive': ['TCF7', 'LEF1', 'SELL', 'CCR7'],
    'cytotoxic': ['NKG7', 'CCL4', 'CST7', 'PRF1', 'GZMA', 'GZMB', 'IFNG', 'CCL3'],
    'exhaustion': ['PDCD1', 'TIGIT', 'LAG3', 'HAVCR2', 'CTLA4'],
    'proliferation': ['MKI67', 'TYMS'],
    'memory': ['AQP3', 'CD69', 'GZMK']
}

ALL = ['nan', '1', '1 or 2', '2', '3', '4', '5', '6', '7']
WOS_PATTERNS = [['nan'], ['1', '1 or 2', '2'], ['3', '4'], ['5', '6', '7']]  # consider ALL + WOS_PATTERNS
PATTERN_NAMES = ['healthy', 'mild', 'moderate', 'severe']  # consider ['all'] + PATTERN_NAMES
CELL_TYPES = [[1, 5], [3, 4], [0, 6], [2]]
CELL_NAMES = ['c1,5_naive', 'c3,4_mem', 'c0,6_eff', 'c2_prof']
LEIDEN2CELLNAME = {
    1: 'c1,5_naive',
    5: 'c1,5_naive',
    3: 'c3,4_mem',
    4: 'c3,4_mem',
    0: 'c0,6_eff',
    6: 'c0,6_eff',
    2: 'c2_prof'
}

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
    'Naive/Memory markers': [
        'CCR7',     # Homing receptor
        'IL7R',     # CD127
    ],
    'Early activation': ['CD69', 'IL2RA'],
    'Proliferation': ['MKI67'],
    'Exhaustion markers': ['PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'CD244', 'ENTPD1'],
    'Terminal differentiation': [
        'KLRG1',    # Terminal differentiation
        'CX3CR1'    # Associated with terminal effectors
    ],
    'TCR signaling': ['ZAP70', 'LAT', 'LCK'],
    'Cytokines': ['IFNG', 'TNF'],
    'Transcription factors': [
        'TOX',      # Associated with exhaustion
        'EOMES',    # Associated with exhaustion/memory
        'TBX21'     # T-bet, associated with effector functions
    ],
    'Metabolic regulators': ['SLC2A1', 'PRKAA1'],
    'Stress response': [
        'HIF1A',    # Hypoxia response
        'XBP1'      # ER stress response
    ],
}
# Define fixed colors for each protein
PROTEIN2COLOR = {
    'Non-structural proteins (NSP)': '#7F63B8',
    'Accessory proteins (ORFs)': '#FF6B6B',
    'Spike (S) protein': '#4ECDC4',
    'Nucleocapsid (N) protein': '#FFD700',
    'Membrane (M) protein': '#4641F0',
    'Envelope (E) protein': '#ED8907',
    'Other': '#9FA4A9'
}


def clean_wos_value(x):
    # Handle floating point nan first
    try:
        if pd.isna(x):  # This handles both string 'nan' and float nan
            return 'nan'
    except:
        pass

    if x == '1 or 2':
        return x

    try:
        return str(int(float(x)))
    except ValueError:
        return x

def clean_wos_get_single_pair(df):
    print("Stringify Who Ordinal Scale..")
    df['Who Ordinal Scale'] = df['Who Ordinal Scale'].apply(clean_wos_value)
    print("Focus on chain_paring == Single pair..")
    df = df[df['chain_pairing'] == 'Single pair']
    print(f"Total number of rows to focus on: {len(df)}")
    return df


def is_valid_peptide(peptide):
    return 8 <= len(peptide) <= 12 and all(c in amino_acids for c in peptide)


def construct_epitope_db(IEDB_db_path, mira_csv_path, outdir, desc="covid19_associated_epitopes",
                         select_valid_peptides=True, expand_peptides=False, target_length=10,
                         max_length_cutoff=None):
    """
    Construct an epitope database from IEDB raw data, merge with MIRA dataset,
    remove redundancies, standardize protein names, and sort by protein frequency.
    This function only concerns COVID-19 antigens. For broader coronavirus, see construct_corona_db().
    NOTE: in the future, remove select_valid_peptides, because database may contain longer epitopes

    Parameters:
    IEDB_db_path (str): Path to the IEDB raw data CSV file.
    mira_csv_path (str): Path to the MIRA dataset CSV file.
    outdir (str): Directory to save the output files.
    desc (str): Description for the output file name (default: "covid19_associated_epitopes").

    Returns:
    pd.DataFrame: The final processed DataFrame.
    """
    # Step 1: Construct the epitope database from IEDB raw data
    df_iedb = pd.read_csv(IEDB_db_path)
    df_iedb = df_iedb.rename(columns={'Epitope - Name': 'peptide', 'Epitope - Source Molecule': 'protein'})
    if is_valid_peptide:
        df = df[df['peptide'].apply(lambda x: is_valid_peptide(x))]
    df_iedb = df_iedb[['peptide', 'protein']]

    # Step 2: Load the MIRA dataset and rename the 'epitope' column to 'peptide'
    df_mira = pd.read_csv(mira_csv_path)
    df_mira = df_mira.rename(columns={'epitope': 'peptide'})

    # Step 3: Append MIRA entries to the IEDB DataFrame
    df_combined = pd.concat([df_iedb, df_mira], ignore_index=True)

    # Step 4: Strip double quotes from the 'protein' column
    df_combined['protein'] = df_combined['protein'].str.strip('"')

    # Step 5: Remove redundant peptides
    df_combined = df_combined.drop_duplicates(subset=['peptide'], keep='first')

    # Step 6: Standardize protein names
    df_combined['protein_standardized'] = df_combined['protein'].apply(standardize_protein_name)

    # Step 7: Calculate protein frequencies and sort by frequency (ascending)
    protein_freq_map = df_combined['protein_standardized'].value_counts().to_dict()
    df_combined['protein_frequency'] = df_combined['protein_standardized'].map(protein_freq_map)
    df_sorted = df_combined.sort_values('protein_frequency', ascending=True)

    # Step 8: Save the final output to a CSV file
    output_file = f"{outdir}/{desc}_merged_sorted.csv"
    df_sorted.to_csv(output_file, index=False)
    print(f"Final database saved as: {output_file}")

    # Step 9: Plot and save the protein frequency histogram
    output_histogram = f"{outdir}/{desc}_protein_histogram.png"
    plt.figure(figsize=(10, 6))
    df_sorted['protein_standardized'].value_counts().plot(kind='bar', color='skyblue')
    plt.xlabel('Protein')
    plt.ylabel('Frequency')
    plt.title('Frequency of COVID-19 Associated Proteins')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_histogram)
    print(f"Protein frequency histogram saved as: {output_histogram}")

    # Step 10: Optionally expand peptides using a sliding window
    if expand_peptides:
        df_expanded = expand_peptides_sliding_window(output_file, outdir, target_length, max_length_cutoff)
        output_file = f'{output}/{desc}_merged_sorted_expanded.csv'
        df_expanded.to_csv(output_file, index=False)
        print(f"Expanded final database saved as: {output_file}")
        return df_expanded
    else:
        return df_sorted

    return df_sorted

def standardize_protein_name(protein_name):
    """
    Standardize protein names into canonical categories.

    Parameters:
    protein_name (str): Original protein name

    Returns:
    str: Standardized protein name
    """
    protein_name = str(protein_name).lower()  # Convert to lowercase for consistent matching

    if any(x in protein_name for x in ['orf', 'replicase', 'polyprotein']):
        return 'ORF1ab protein'
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


def expand_peptides_sliding_window(input_csv, outdir, target_length=10, max_length_cutoff=None):
    """
    Reads a CSV file containing peptide sequences, expands sequences longer than the target length
    into multiple sequences using a sliding window approach.

    Args:
        input_csv (str): Path to input CSV file
        outdir (str): Directory to save the output CSV
        target_length (int): Desired length for peptide sequences (default: 10)
        max_length_cutoff (int): Optional maximum length cutoff. Peptides longer than this will be expanded.
                                If None, will use target_length as cutoff.
    """
    # Set max_length_cutoff to target_length if not specified
    if max_length_cutoff is None:
        max_length_cutoff = target_length

    # Input validation
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if max_length_cutoff < target_length:
        raise ValueError("max_length_cutoff must be greater than or equal to target_length")

    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Initialize list to store new rows
    expanded_rows = []

    # Process each row
    for _, row in df.iterrows():
        peptide = row['peptide']
        protein = row['protein']

        # If peptide length is <= max_length_cutoff, keep as is
        if len(peptide) <= max_length_cutoff:
            expanded_rows.append({'peptide': peptide, 'protein': protein})
        else:
            # Create sliding windows of target_length
            for i in range(len(peptide) - target_length + 1):
                peptide_window = peptide[i:i+target_length]
                expanded_rows.append({'peptide': peptide_window, 'protein': protein})

    # Create new DataFrame from expanded rows
    df_expanded = pd.DataFrame(expanded_rows)

    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    # Generate output filename based on parameters
    output_filename = f'expanded_length_{target_length}.csv'
    output_path = os.path.join(outdir, output_filename)

    # Save to CSV
    df_expanded.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)

    return df_expanded


def construct_corona_db(IEDB_db_path, covid19_epitopes, outdir, desc, window_size=10):
    """
    Construct a coronavirus epitope database from IEDB raw data, remove COVID-19 specific epitopes,
    and expand peptides longer than the specified window size using a sliding window.

    Parameters:
    IEDB_db_path (str): Path to the IEDB raw data CSV file.
    covid19_epitopes (str): Path to the COVID-19 specific epitopes CSV file.
    outdir (str): Directory to save the output files.
    desc (str): Description for the output file name.
    window_size (int): Desired length for peptide sequences when expanding (default: 10).

    Returns:
    pd.DataFrame: The final processed DataFrame.
    """
    # Step 1: Load the IEDB database and rename columns
    df = pd.read_csv(IEDB_db_path)
    df = df.rename(columns={'Epitope - Name': 'peptide', 'Epitope - Source Molecule': 'protein'})

    # Step 2: Remove redundant peptides by keeping only unique peptides
    df = df.drop_duplicates(subset=['peptide'])

    # Step 3: Remove COVID-19 specific epitopes
    df_covid19 = pd.read_csv(covid19_epitopes)
    df = df[~df['peptide'].isin(df_covid19['peptide'])]

    # Save the unexpanded database to a CSV file
    Path(outdir).mkdir(parents=True, exist_ok=True)
    unexpanded_output_file = f"{outdir}/{desc}.csv"
    df.to_csv(unexpanded_output_file, index=False)
    print(f"Unexpanded database saved as: {unexpanded_output_file}")

    # Step 4: Expand peptides longer than the window size using a sliding window
    df_expanded = expand_peptides_sliding_window(unexpanded_output_file, outdir, target_length=window_size, max_length_cutoff=window_size)

    # Step 5: Save the expanded database to a CSV file
    expanded_output_file = f"{outdir}/{desc}_expanded.csv"
    df_expanded.to_csv(expanded_output_file, index=False)
    print(f"Expanded database saved as: {expanded_output_file}")

    return df_expanded


def read_gex_file(file_path):
    """
    Read a single GEX file and return a DataFrame with additional metadata.
    """
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    df['source_file'] = file_path
    df['cell_barcode'] = df.index
    return df

def read_all_data(data_dir, gex_cache=None, obs_cache=None, use_multiprocessing=False, n_processes=None, filtering=False):
    """
    Read all data. Basically, load gene expression data and the metadata (observation).
    When operating first, use the `filtering` argument to only get the CD8+ T cells information.
    After, load that dataset using `gex_cache` for the subsequent analysis.
    This is because EpiGen only works with CD8+ T cells.

    Parameters
    ----------
    data_dir: str
        Directory where covid19 data were downloaded
    gex_cache:
        The cached gene expression data to load from (gene expression cache)
    obs_cache: str
        The path to the annotated data in csv format (observation cache)
    use_multiprocessing: bool
        Useful when obs_cache is None, that is when reading from independent gex txt files
    n_processes: int / None
        Number of processes to use (when use_multiprocessing is True)
    filtering: bool
        A crucial step to retrieve CD8+ T cells from GEX datasets using the cell barcode in the TCR dataset
    """
    if gex_cache is None:
        file_pattern = 'gex'
        gex_files = [f for f in os.listdir(data_dir) if file_pattern in f.lower()]

        if not gex_files:
            raise ValueError(f"No files containing '{file_pattern}' found in {data_dir}")

        file_paths = [os.path.join(data_dir, f) for f in gex_files]

        if use_multiprocessing:
            if n_processes is None:
                n_processes = int(mp.cpu_count() * 0.8)
            with mp.Pool(processes=n_processes) as pool:
                results = list(tqdm(pool.imap(read_gex_file, file_paths),
                                    total=len(file_paths), desc="Processing GEX files"))
        else:
            results = [read_gex_file(fp) for fp in tqdm(file_paths, desc="Processing GEX files")]

        # Filter out None results (failed reads)
        dfs = [df for df in results if df is not None]

        # Concatenate all DataFrames
        combined_df = pd.concat(dfs, axis=0, sort=False)

        # Create AnnData object
        adata = sc.AnnData(combined_df.drop(['source_file', 'cell_barcode'], axis=1))
        adata.obs['source_file'] = combined_df['source_file']
        adata.obs_names = combined_df['cell_barcode']
        print(f"Combined data shape: {adata.shape}")

        # # Preprocessing
        # sc.pp.normalize_total(adata, target_sum=1e6)  # CPM normalization
        # sc.pp.log1p(adata)  # log1p transformation

        # # Highly variable genes
        # sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        # breakpoint()
    else:
        print(f"Read from {gex_cache}..")
        adata = ad.read_h5ad(gex_cache)
        print(f"adata shape: {adata.shape}")

    if obs_cache:
        print(f"Reading cache from {obs_cache}..")
        df_cache = pd.read_csv(obs_cache)

        # Ensure that df_cache has the same index as mdata['gex'].obs
        df_cache.index = adata.obs.index

        # Assign df_cache back to mdata['gex'].obs
        adata.obs = df_cache

    if filtering:
        print("Read CD8 TCR data to get the cell barcodes of CD8+ T cells..")
        tcr_files = []
        for file in os.listdir(data_dir):
            if 'cd8' in file:
                file_path = os.path.join(data_dir, file)
                df = pd.read_csv(file_path, delimiter='\t')
                tcr_files.append(df)

        # Concatenate all DataFrames and reset the index
        df_tcr = pd.concat(tcr_files, ignore_index=True)

        # Ensure the index of adata.obs is named 'cell_barcode'
        adata.obs = adata.obs.reset_index()

        # Rename 'Unnamed: 0' to 'cell_barcode' in df_tcr for clarity
        df_tcr = df_tcr.rename(columns={'Unnamed: 0': 'cell_barcode'})

        # Merge df_tcr with adata.obs
        merged_df = pd.merge(df_tcr, adata.obs, on='cell_barcode', how='inner')

        # If you want to create a new AnnData object with only the matched cells:
        # adata_matched = ad.AnnData(adata[merged_df.cell_barcode].X, obs=merged_df)
        adata_matched = adata[adata.obs['cell_barcode'].isin(merged_df['cell_barcode'])]
        adata_matched.obs_names = adata_matched.obs['cell_barcode']
        Path('gex_cache').mkdir(parents=True, exist_ok=True)
        adata_matched.write('gex_cache/cd8_gex.h5ad')
        print("gex_cache/cd8_gex.h5ad was saved. ")
        return adata_matched

    return adata


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
    df1 = df1.dropna(subset=['TRB_1_cdr3'])
    df2 = df2.dropna(subset=['TRB_1_cdr3'])
    df1 = df1[df1['chain_pairing'] == 'Single pair']
    df2 = df2[df2['chain_pairing'] == 'Single pair']

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
    matched_tcrs_1 = set(df1_matched['TRB_1_cdr3'])
    matched_tcrs_2 = set(df2_matched['TRB_1_cdr3'])

    # Calculate overlap
    overlap = matched_tcrs_1 & matched_tcrs_2

    # Calculate expected random overlap percentage
    total_tcrs = len(df1)
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
        overlap_pct_1 = (len(df1_matched[df1_matched['TRB_1_cdr3'].isin(overlap)]) / len(df1_matched)) * 100
        print(f"- Percentage of File 1 matches that overlap: {overlap_pct_1:.2f}%")
    else:
        print("- No matches in File 1")

    if len(df2_matched) > 0:
        overlap_pct_2 = (len(df2_matched[df2_matched['TRB_1_cdr3'].isin(overlap)]) / len(df2_matched)) * 100
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


def ensemble_PA_marked(PA_marked_list, outdir, th=0.5, desc='PA', col='match'):
    """
    Ensemble multiple PA_marked.csv files using majority voting based on cumulative matches
    at specific k-mer cutoffs (1, 2, 4, 8, 16, 32).
    Args:
        PA_marked_list (list): List of paths to PA_marked csv files
        outdir (str): Output directory path
        th (float): Threshold for majority voting (default: 0.5)
    """
    cutoff_indices = list(range(32))

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


def inspect_num_PA(PA_marked, col='match'):
    df = pd.read_csv(PA_marked)
    df = df[df['chain_pairing'] == 'Single pair']
    for k in [1, 2, 4, 8, 16, 32]:
        match_cols = [f'{col}_{i}' for i in range(k)]
        N = len(df[df[match_cols].ge(1).any(axis=1)])
        print(f"{PA_marked} - k: {k}, N: {N}")


def get_stars(pvalue):
    if pvalue <= 0.001:
        return '***'
    elif pvalue <= 0.01:
        return '**'
    elif pvalue <= 0.05:
        return '*'
    return 'ns'


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
    df = clean_wos_get_single_pair(df)
    print(f"Total of {len(df)} entries.")

    # Define pattern names
    pattern_names = ['all'] + PATTERN_NAMES
    wos_patterns = [ALL] + WOS_PATTERNS

    # Initialize a nested dictionary to store the counts
    num_samples = {pattern_name: {cell_type: 0 for cell_type in CELL_NAMES} for pattern_name in pattern_names}

    # Calculate the number of samples for each cell type and pattern name
    if check_PA:
        match_columns = [f'match_{i}' for i in range(K)]

    for celltype, cell_name in zip(CELL_TYPES, CELL_NAMES):
        for wos_pattern, pattern_name in zip(wos_patterns, pattern_names):
            if check_PA:
                num = len(df[(df['Who Ordinal Scale'].isin(wos_pattern)) & (df['leiden'].isin(celltype)) & (df[match_columns].any(axis=1))])
            else:
                num = len(df[(df['Who Ordinal Scale'].isin(wos_pattern)) & (df['leiden'].isin(celltype))])
            num_samples[pattern_name][cell_name] = num

    # Print the results based on the specified hierarchy
    if check_PA:
        print("We focus on PA T cells, because check_PA=True")

    if hierarchy == 'pattern':
        for pattern_name in pattern_names:
            print(f"\nPattern: {pattern_name}")
            for cell_name in CELL_NAMES:
                print(f"  Cell Type: {cell_name}, Number of Samples: {num_samples[pattern_name][cell_name]}")
    elif hierarchy == 'celltype':
        for cell_name in CELL_NAMES:
            print(f"\nCell Type: {cell_name}")
            for pattern_name in pattern_names:
                print(f"  Pattern: {pattern_name}, Number of Samples: {num_samples[pattern_name][cell_name]}")
    else:
        raise ValueError("Invalid hierarchy argument. Use 'pattern' or 'celltype'.")
