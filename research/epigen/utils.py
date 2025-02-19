import os
import re
import time
import pickle
import random
import argparse
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mhcgnomes
from sklearn.model_selection import train_test_split
from functools import wraps
from scipy.stats import skew, kurtosis
from functools import partial
from tqdm import tqdm
from pathlib import Path
os.environ['NUMBA_CACHE_DIR'] = '__pycache__'
import Levenshtein
from concurrent.futures import ProcessPoolExecutor

sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 20, 'linewidths':0}
# os.environ['NUMBA_DISABLE_JIT'] = 1


amino_acids = set('ACDEFGHIKLMNPQRSTVWY')

aa_to_index = {
    'A': 0,
    'R': 1,
    'N': 2,
    'D': 3,
    'C': 4,
    'Q': 5,
    'E': 6,
    'G': 7,
    'H': 8,
    'I': 9,
    'L': 10,
    'K': 11,
    'M': 12,
    'F': 13,
    'P': 14,
    'S': 15,
    'T': 16,
    'W': 17,
    'Y': 18,
    'V': 19,
    'X': 20
}

index_to_aa = {
    0 : 'A',
    1 : 'R',
    2 : 'N',
    3 : 'D',
    4 : 'C',
    5 : 'Q',
    6 : 'E',
    7 : 'G',
    8 : 'H',
    9 : 'I',
    10: 'L',
    11: 'K',
    12: 'M',
    13: 'F',
    14: 'P',
    15: 'S',
    16: 'T',
    17: 'W',
    18: 'Y',
    19: 'V',
    20: 'X',
    21: "",
    22: "",
    23: ""
}

def is_valid_cdr(cdr):
    return 10 <= len(cdr) <= 20 and all(c in amino_acids for c in cdr)

def is_valid_peptide(peptide):
    return 8 <= len(peptide) <= 12 and all(c in amino_acids for c in peptide)

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} takes {end_time - start_time} sec")
        return result
    return wrapper


def adheres_to_vocab(s: str, vocab) -> bool:
    """
    Returns whether a given string contains only characters from vocab
    >>> adheres_to_vocab("RKDES")
    True
    >>> adheres_to_vocab(AMINO_ACIDS + AMINO_ACIDS)
    True
    """
    return set(s).issubset(set(vocab))


def split_data_for_bootstrapping(data_path, n_samples, output_dir):
    """
    Generates bootstrap samples from the dataset and saves them as CSV files.
    This is used to train Robust Affinity Predictor.

    Parameters
    ----------
    data_path: str
        path to csv
    n_samples: int
        The number of bootstrap samples to generate.
    output_dir: str
        The directory to save the bootstrap sample CSV files.

    Returns:
    --------
    list: A list of filenames, where each filename corresponds to a saved bootstrap sample CSV file.
    """
    df = pd.read_csv(data_path)
    train_df = df[df['train_test'] == 'train']
    test_df = df[df['train_test'] == 'test']

    bag_of_data = []
    filenames = []
    n = len(train_df)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(n_samples):
        # Sample with replacement from the training subset
        train_sample = train_df.sample(n=n, replace=True)

        # Combine the sampled training data with the unchanged test data
        combined_sample = pd.concat([train_sample, test_df])

        # Generate a filename for the sample
        filename = f"bootstrap_sample_{i+1}.csv"
        filepath = os.path.join(output_dir, filename)

        # Save the sample to a CSV file
        combined_sample.to_csv(filepath, index=False)

        # Append the filename to the list
        filenames.append(filepath)
        bag_of_data.append(combined_sample)

    return filenames


def postprocess_sampled_data(root, keyword, outdir="sampled_data", topk=None, use_mhc=False):
    """
    After sampling the high affinity data using ActiveSampler,
    integrate them and convert to a format that can be used
    for training. Depending on the 'partial' flag, save the data
    as separate CSV files for each folder or as a single CSV file
    for all folders.
    """
    print(f"postprocess_sampled_data: topk={topk}")

    folders = [os.path.join(root, x) for x in os.listdir(root) if keyword in x]
    Path(outdir).mkdir(parents=True, exist_ok=True)

    all_data = []

    def process_tcr_summary(tcr, summary, topk, use_mhc):
        if use_mhc:
            epi_mhc = summary[tcr]['epitopes']
            epitopes = [x.split("_")[0] for x in epi_mhc]
            mhcs = [x.split("_")[1] for x in epi_mhc]
            skewness = summary[tcr]['skewness']

            if topk:
                return [(tcr, mhcs[i], epitopes[i]) for i in range(topk)]
            else:
                choose = 1 if skewness < 2.0 else int(min(skewness * 10, 5))
                return [(tcr, mhcs[i], epitopes[i]) for i in range(choose)]
        else:
            epitopes = summary[tcr]
            return [(tcr, epitopes[i]) for i in range(topk)]

    for folder in tqdm(folders, desc='Processing folders'):
        pkls = [os.path.join(folder, y) for y in os.listdir(folder) if y.endswith(".pkl")]

        for pkl in tqdm(pkls, desc='Processing pkl files'):
            with open(pkl, 'rb') as f:
                summary = pickle.load(f)

            for tcr in summary.keys():
                all_data.extend(process_tcr_summary(tcr, summary, topk, use_mhc))

    columns = ['tcr', 'mhc', 'epitope'] if use_mhc else ['tcr', 'epitope']
    df = pd.DataFrame(all_data, columns=columns)
    df.to_csv(f"{outdir}/all_data_topk{topk}.csv", index=False)


def remove_redundancy(data_csv, th):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(data_csv)

    # Count the frequency of each peptide
    peptide_counts = df['epitope'].value_counts()
    folder = Path(data_csv).parent
    name = Path(data_csv).stem

    if isinstance(th, list):
        outfiles = []
        for th_single in th:
            df_filtered = df[df['epitope'].map(peptide_counts) <= th_single]
            # Construct the output file path
            outfile = folder / f"{name}_th_{th_single}.csv"

            # Save the filtered DataFrame to a new CSV file
            df_filtered.to_csv(outfile, index=False)
            print(f"{outfile} was saved.")
            outfiles.append(outfile)
        return outfiles
    else:
        # Filter the DataFrame to keep only rows with peptide frequency less than or equal to th
        df_filtered = df[df['epitope'].map(peptide_counts) <= th]

        # Construct the output file path
        outfile = folder / f"{name}_th_{th}.csv"

        # Save the filtered DataFrame to a new CSV file
        df_filtered.to_csv(outfile, index=False)
        print(f"{outfile} was saved.")

        return outfile


def format_sampled_data(sampled_data_csv, outfile="all_data_gpt2.csv", use_mhc=False):
    """
    Merge (tcr, peptide) info with mhc sequence
    """
    df = pd.read_csv(sampled_data_csv)
    if use_mhc:
        df['text'] = df['tcr'] + '|' + df['mhc']
        df['label'] = df['epitope']
    else:
        df['text'] = df['tcr']
        df['label'] = df['epitope']
    final_df = df[['text', 'label']]
    final_df.to_csv(outfile, index=False)
    print(f"{outfile} was saved. ")
    return outfile


def convert_pred_to_tcrmodel2_format(pred_csv, peptide_db, pseudo2full_pkl, tcr_alpha_template, tcr_beta_template, pseudo,
                                     use_mhc=False, epitope_col='pred_0', num_controls=20, num_samples=20):
    dataset = Path(pred_csv).stem.split("_")[1]
    df_pred = pd.read_csv(pred_csv)
    print("Reading the peptide DB..")
    peps = pd.read_csv(peptide_db)
    peptide_pool = list(set(peps['peptide'].tolist()))
    random.shuffle(peptide_pool)
    peptide_pool = peptide_pool[:100000]
    print("Peptide DB reading complete!")

    with open(pseudo2full_pkl, 'rb') as f:
        pseudo2full = pickle.load(f)
    mhca_seq = pseudo2full.get(pseudo, "Unknown MHC sequence")

    outputs = []

    def create_tcrmodel2_args(cdr3, pseudo, epitope, tcr_beta_grafted):
        return {
            'tcra_seq': tcr_alpha_template,
            'tcrb_seq': tcr_beta_grafted,
            'pep_seq': epitope,
            'mhca_seq': pseudo2full.get(pseudo, "Unknown MHC sequence"),
            'tcr': cdr3,
            'mhc': pseudo,
            'epitope': epitope
        }

    print("Constructing TCRmodel2 args..")
    print(f"use_mhc: {use_mhc}, epitope_col: {epitope_col}, num_controls: {num_controls}\nnum_samples=20")
    df_pred = df_pred.sample(n=num_samples)
    for _, row in tqdm(df_pred.iterrows()):
        cdr3 = row['tcr']
        pseudo = row['mhc'] if use_mhc else pseudo
        epitope = row[epitope_col]
        # Graft CDR3 into beta template
        tcr_beta_grafted = tcr_beta_template[:90] + cdr3 + tcr_beta_template[90 + 15:]  # 15 is the length of the CDR3 region of this template
        # Create and append main tcrmodel2 args
        outputs.append(create_tcrmodel2_args(cdr3, pseudo, epitope, tcr_beta_grafted))
        # Create and append control tcrmodel2 args
        for epi in random.sample(peptide_pool, num_controls):
            outputs.append(create_tcrmodel2_args(cdr3, pseudo, epi, tcr_beta_grafted))
    # Save the outputs to a file
    output_path = os.path.join(os.path.dirname(pred_csv), f'tcrmodel2_args_{dataset}_{epitope_col}.pkl')
    with open(output_path, 'wb') as out_file:
        pickle.dump(outputs, out_file)
    return outputs


def convert_pred_to_tcrmodel2_format_continual(pred_csv, tcrmodel2_arg_ref, desc, epitope_col='pred_0'):
    print(f"Converting to the tcrmodel2 args based on the reference: {tcrmodel2_arg_ref}")
    df_pred = pd.read_csv(pred_csv)
    with open(tcrmodel2_arg_ref, "rb") as f:
        ref = pickle.load(f)
    print("Predictions and reference were successfully read!")

    # Create a dictionary for faster lookup
    tcr_to_peptide = dict(zip(df_pred['tcr'], df_pred[epitope_col]))

    outputs = []
    for item in ref[::21]:  # Select every 21st item
        tcr = item['tcr']
        if tcr in tcr_to_peptide:
            peptide = tcr_to_peptide[tcr]
            item['pep_seq'] = peptide
            item['epitope'] = peptide
            outputs.append(item)
        else:
            print(f"Warning: TCR {tcr} not found in predictions.")

    # Save the outputs to a file
    output_path = os.path.join(os.path.dirname(pred_csv), f'tcrmodel2_args_{desc}.pkl')
    with open(output_path, 'wb') as out_file:
        pickle.dump(outputs, out_file)
    print(output_path)

    return outputs


def format_test_set(test_csv, mhc_allele_info, output_csv='formatted_test_set.csv'):
    """
    Format the test set S1, S2, S3, and S4 of TABR-BERT
    """
    # Load the test set and auxiliary MHC allele information
    df_test = pd.read_csv(test_csv)
    df_mhc = pd.read_csv(mhc_allele_info)

    # Keep only the positive labels if needed
    df_test = df_test[df_test['label'] == 1]

    # Rename columns to facilitate merging
    df_mhc.rename(columns={'allele': 'allele', 'sequence': 'mhc'}, inplace=True)

    # Merge to map allele to MHC pseudo sequence
    df_merged = pd.merge(df_test, df_mhc, on='allele', how='left')

    # Create 'text' column with "tcr|mhc" format
    df_merged['text'] = df_merged['cdr3'] + '|' + df_merged['mhc']

    # Select and rename the columns to match the desired output format
    df_result = df_merged[['text', 'peptide']].rename(columns={'peptide': 'label'})

    # Save the formatted data to a CSV file
    df_result.to_csv(output_csv, index=False)

    return df_result


def split_data(file_path, random_seed=None):
    data = pd.read_csv(file_path)

    # Split the dataset into training (70%) and the rest (30%)
    train_set, temp_set = train_test_split(data, test_size=0.2, random_state=random_seed)

    # Split the remaining data into validation (20% of the total) and test (10% of the total)
    val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=random_seed)

    # Print the shapes of the datasets
    print("Training set shape:", train_set.shape)
    print("Validation set shape:", val_set.shape)
    print("Test set shape:", test_set.shape)

    # Save the splits to CSV files
    name = Path(file_path).stem
    folder = Path(file_path).parent
    train_set.to_csv(folder / f"{name}_train.csv", index=False)
    val_set.to_csv(folder / f"{name}_val.csv", index=False)
    test_set.to_csv(folder / f"{name}_test.csv", index=False)

    return train_set, val_set, test_set


def merge_data(files, split, outdir):
    # Merge train sets from multiple subsets into one
    assert split in ['train', 'val', 'test']
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs)
    outfile = f"{outdir}/{split}.csv"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df.to_csv(outfile, index=False)
    print(f"{outfile} was saved. ")


def find_unseen_tcr_and_entity(train_csv, test_csv, entity='mhc', outdir="data"):
    """
    Splits the test set into 4 mutually exclusive subsets based on whether either TCR or the specified entity (MHC or peptide)
    were seen at training time, and saves these subsets as CSV files.
    Parameters
    ----------
    train_csv : str
        Path to the training data CSV file.
    test_csv : str
        Path to the test data CSV file.
    entity : str, optional
        The entity to consider for splitting the data, either 'mhc' or 'pep' (default is 'mhc').
    outdir : str, optional
        The directory where the output CSV files will be saved (default is "data").
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    # Read the training and test data
    df_tr = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # Extract entity and TCR information
    if entity == 'mhc':
        df_tr['entity'] = df_tr['text'].apply(lambda x: x.split('|')[1])
        df_test['entity'] = df_test['text'].apply(lambda x: x.split('|')[1])
    elif entity == 'pep':
        df_tr['entity'] = df_tr['label']
        df_test['entity'] = df_test['label']
    df_tr['tcr'] = df_tr['text']
    df_test['tcr'] = df_test['text']

    # Find unseen entities and TCRs in the test set
    unseen_entity = set(df_test['entity']) - set(df_tr['entity'])
    unseen_tcr = set(df_test['tcr']) - set(df_tr['tcr'])

    # Create mutually exclusive subsets
    df_test_unseen_both = df_test[df_test['entity'].isin(unseen_entity) & df_test['tcr'].isin(unseen_tcr)]
    df_test_unseen_entity = df_test[df_test['entity'].isin(unseen_entity) & ~df_test['tcr'].isin(unseen_tcr)]
    df_test_unseen_tcr = df_test[~df_test['entity'].isin(unseen_entity) & df_test['tcr'].isin(unseen_tcr)]
    df_test_seen_both = df_test[~df_test['entity'].isin(unseen_entity) & ~df_test['tcr'].isin(unseen_tcr)]

    # Save subsets as CSV files
    df_test_unseen_both[['text', 'label']].to_csv(f"{outdir}/test_unseen_both.csv", index=False)
    df_test_unseen_entity[['text', 'label']].to_csv(f"{outdir}/test_unseen_{entity}.csv", index=False)
    df_test_unseen_tcr[['text', 'label']].to_csv(f"{outdir}/test_unseen_tcr.csv", index=False)
    df_test_seen_both[['text', 'label']].to_csv(f"{outdir}/test_seen_both.csv", index=False)

    # Verify mutual exclusiveness and total length
    total_length = len(df_test)
    sum_subset_lengths = len(df_test_unseen_both) + len(df_test_unseen_entity) + len(df_test_unseen_tcr) + len(df_test_seen_both)

    assert sum_subset_lengths == total_length, "Sum of subset lengths does not match total test set length"

    print(f"Total test set length: {total_length}")
    print(f"Unseen both: {len(df_test_unseen_both)}")
    print(f"Unseen {entity}: {len(df_test_unseen_entity)}")
    print(f"Unseen TCR: {len(df_test_unseen_tcr)}")
    print(f"Seen both: {len(df_test_seen_both)}")
    print(f"Sum of subsets: {sum_subset_lengths}")


def draw_learning_curve(outdir, records, descs):
    """
    After evaluating using run_clm_predict.py, a file containing metrics per epoch
    is created. This function plots the learning curve based on this data.
    """
    data = []
    for record in records:
        data.append(pd.read_csv(record, names=['epoch', 'acc', 'loss', 'perplexity']))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    for df, desc in zip(data, descs):
        ax1.plot(df['epoch'], df['acc'], label=desc)
        ax2.plot(df['epoch'], df['loss'], label=desc)
        ax3.plot(df['epoch'], df['perplexity'], label=desc)

    ax1.set_title("Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    ax3.set_title("Validation Perplexity")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Perplexity")
    ax3.legend()

    plt.tight_layout()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{outdir}/learning_curve.pdf", format='pdf')
    print(f"{outdir}/learning_curve.pdf was saved.")


def get_dataset_stat(data_csv):
    df = pd.read_csv(data_csv)

    # Unique TCRs and epitopes
    try:
        unique_tcrs = df['tcr'].unique()
        unique_epitopes = df['epitope'].unique()
    except:
        unique_tcrs = df['text'].unique()
        unique_epitopes = df['label'].unique()

    # Total number of rows
    total_rows = len(df)

    # TCR to peptide ratio
    tcr2peptide_ratio = len(unique_tcrs) / len(unique_epitopes)

    # Print the information
    print(data_csv)
    print(f"Total rows: {total_rows}")
    print(f"Unique TCRs: {len(unique_tcrs)}")
    print(f"Unique Epitopes: {len(unique_epitopes)}")
    print(f"TCR to Peptide Ratio: {tcr2peptide_ratio:.2f}")


def control_pseudo_labeled_set_ratio(outdir, public_csv, pseudo_csv, ratio=1.00):
    """Control the ratio of pseudo-labeled dataset while keeping full public dataset.

    Parameters
    ----------
    outdir : str
        Output directory path for saving the combined dataset
    public_csv : str
        Path to the public dataset CSV file
    pseudo_csv : str
        Path to the pseudo-labeled dataset CSV file
    ratio : float, optional
        Fraction of pseudo-labeled dataset to include, by default 1.00

    Returns
    -------
    pandas.DataFrame
        Combined dataset with specified ratio of pseudo-labeled data

    Notes
    -----
    The function always uses 100% of the public dataset and controls the size of
    the pseudo-labeled dataset using the ratio parameter. This is useful for
    testing the effect of pseudo-labeled dataset size.
    """
    # Read datasets
    df_pub = pd.read_csv(public_csv)
    df_pseudo = pd.read_csv(pseudo_csv)

    # Rename columns in public dataset if needed
    if 'tcr' in df_pub.columns and 'epitope' in df_pub.columns:
        df_pub = df_pub.rename(columns={
            'tcr': 'text',
            'epitope': 'label'
        })
    # Rename columns in public dataset if needed
    if 'tcr' in df_pseudo.columns and 'epitope' in df_pseudo.columns:
        df_pseudo = df_pseudo.rename(columns={
            'tcr': 'text',
            'epitope': 'label'
        })

    # Sample from pseudo-labeled dataset
    if ratio < 1.0:
        df_pseudo = df_pseudo.sample(frac=ratio, random_state=42)

    # Concatenate datasets
    combined_df = pd.concat([df_pub, df_pseudo], ignore_index=True)

    # Save the combined dataset
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    output_file = outdir_path / f"{Path(pseudo_csv).stem}_ratio_{ratio:.2f}.csv"
    combined_df.to_csv(output_file, index=False)

    print(f"Saved combined dataset with {len(df_pub)} public and {len(df_pseudo)} pseudo samples")

    return combined_df


def check_dataset_overlap(query, reference):
    """
    Check overlap between query sequences and reference TCRs/epitopes.

    Parameters:
    query (str): Path to query CSV file with sequences
    reference (str): Path to reference CSV file with TCRs and epitopes

    Returns:
    tuple: (DataFrame with match results, dict with overlap statistics)
    """
    # Read input files
    df_q = pd.read_csv(query, names=['sequence'])
    df_r = pd.read_csv(reference)

    # Create sets of reference TCRs and epitopes
    tcrs = set(df_r['tcr'].unique())
    epitopes = set(df_r['epitope'].unique())

    # Combine TCRs and epitopes into one set for matching
    reference_set = tcrs.union(epitopes)

    # Check inclusion of query sequences in reference set
    df_q['match'] = df_q['sequence'].isin(reference_set)

    # Calculate overlap statistics
    stats = {
        'total_queries': len(df_q),
        'matching_sequences': df_q['match'].sum(),
        'unique_queries': df_q['sequence'].nunique(),
        'overlap_percentage': (df_q['match'].sum() / len(df_q) * 100),
        'reference_tcrs': len(tcrs),
        'reference_epitopes': len(epitopes)
    }
    print(stats)
    return df_q, stats


def construct_epitope_db(IEDB_db_path, outdir, desc,
                         expand_peptides=True, target_length=10, max_length_cutoff=None):
    """
    Construct an epitope database from IEDB raw data
    remove redundancies, and sort by protein frequency.

    Parameters:
    IEDB_db_path (str): Path to the IEDB raw data CSV file.
    outdir (str): Directory to save the output files.
    desc (str): Description for the output file name (default: "covid19_associated_epitopes").

    Returns:
    pd.DataFrame: The final processed DataFrame.
    """
    # Step 1: Construct the epitope database from IEDB raw data
    df_iedb = pd.read_csv(IEDB_db_path)
    df_iedb = df_iedb.rename(columns={'Epitope - Name': 'peptide', 'Epitope - Source Molecule': 'protein'})
    # df = df[df['peptide'].apply(lambda x: is_valid_peptide(x))]
    df_iedb = df_iedb[['peptide', 'protein']]

    # Step 3: Append MIRA entries to the IEDB DataFrame
    df_combined = df_iedb

    # Step 4: Strip double quotes from the 'protein' column
    df_combined['protein'] = df_combined['protein'].str.strip('"')

    # Step 5: Remove redundant peptides
    df_combined = df_combined.drop_duplicates(subset=['peptide'], keep='first')

    # Step 6: Standardize protein names
    # df_combined['protein_standardized'] = df_combined['protein'].apply(standardize_protein_name)

    # Step 7: Calculate protein frequencies and sort by frequency (ascending)
    protein_freq_map = df_combined['protein'].value_counts().to_dict()
    df_combined['protein_frequency'] = df_combined['protein'].map(protein_freq_map)
    df_sorted = df_combined.sort_values('protein_frequency', ascending=True)

    # Step 8: Save the final output to a CSV file
    output_file = f"{outdir}/{desc}_sorted.csv"
    df_sorted.to_csv(output_file, index=False)
    print(f"Final database saved as: {output_file}")

    # Step 10: Optionally expand peptides using a sliding window
    if expand_peptides:
        df_expanded = expand_peptides_sliding_window(output_file, outdir, target_length, max_length_cutoff)
        output_file = f'{outdir}/{desc}_sorted_expanded.csv'
        df_expanded.to_csv(output_file, index=False)
        print(f"Expanded final database saved as: {output_file}")
        return df_expanded
    else:
        return df_sorted

    return df_sorted

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
    # os.makedirs(outdir, exist_ok=True)

    # Generate output filename based on parameters
    # output_filename = f'expanded_length_{target_length}.csv'
    # output_path = os.path.join(outdir, output_filename)

    # Save to CSV
    # df_expanded.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)

    return df_expanded
