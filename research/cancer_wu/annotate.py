import os
import pandas as pd
import numpy as np
import muon as mu

import matplotlib.pyplot as plt
import scirpy as ir
import mudata as md
import Levenshtein
from multiprocessing import Pool, cpu_count
from pathlib import Path

from research.cancer_wu.utils import read_all_data


def annotation_wrapper(data_dir, pred_csv=None, epi_db_path=None, obs_cache=None):
    # A wrapper function to run series of annotations
    mdata = read_all_data(data_dir, obs_cache=obs_cache, filter_cdr3_notna=False, filter_cell_types=False, cdr_annotation_path=None)
    mdata = annotate_cdr3_scirpy(mdata)
    save_cdr3_formatted(mdata, outdir="cancer_wu/obs_cache")
    return
    mdata = insert_epitope_info(mdata, pred_csv)
    mdata = annotate_tumor_associated_epitopes(mdata, epi_db_path, method='substring')
    mdata = annotate_sites(mdata)
    return mdata


def annotate_cdr3_scirpy(mdata, outdir=None):
    print("read_all_data(): Annotate CDR3b and CDR3b_nt..")
    ### Use scirpy for QC
    ir.pp.index_chains(mdata)
    ir.tl.chain_qc(mdata)
    mu.pp.filter_obs(mdata, "airr:chain_pairing", lambda x: np.isin(x, ["single pair", "orphan VDJ", "extra VJ", "extra VDJ"]))
    # Get the CDR3b sequences
    cell_id2cdr3 = {}
    cell_id2cdr3_nt = {}
    for i in range(len(mdata['airr'])):
        cell_id = mdata['airr'].obs['new_cell_id'][i]
        loci = list(mdata['airr'].obsm['airr'].locus[i])
        seqs = list(mdata['airr'].obsm['airr'].junction_aa[i])
        seqs_nt = list(mdata['airr'].obsm['airr'].junction[i])
        for locus, seq, seq_nt in zip(loci, seqs, seqs_nt):
            if locus == 'TRB' and seq is not None:
                cell_id2cdr3[cell_id] = seq
                cell_id2cdr3_nt[cell_id] = seq_nt
                break
    df_cdr3 = pd.DataFrame(list(cell_id2cdr3.items()), columns=['cell_id', 'cdr3'])
    df_cdr3_nt = pd.DataFrame(list(cell_id2cdr3_nt.items()), columns=['cell_id', 'cdr3_nt'])
    # Merge the CDR3b sequence to GEX obs by cell_id
    mdata = read_all_data("data/raw")
    obs_df = mdata['gex'].obs
    merged_df = obs_df.merge(df_cdr3, how='left', left_index=True, right_on='cell_id')
    # Set the index of the merged DataFrame to the original index (cell_id)
    merged_df.set_index('cell_id', inplace=True)
    merged_df = merged_df.merge(df_cdr3_nt, how='left', left_index=True, right_on='cell_id')
    # Set the index of the merged DataFrame to the original index (cell_id)
    merged_df.set_index('cell_id', inplace=True)
    # Assign the merged DataFrame back to mdata['gex'].obs
    mdata['gex'].obs = merged_df
    if outdir is None:
        outdir = "gex_obs"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    mdata['gex'].obs.to_csv(f"{outdir}/cdr3_added_scirpy.csv", index=True)
    print(f"{outdir}/cdr3_added_scirpy.csv was saved. ")
    return mdata


def save_cdr3_formatted(mdata, outdir=None):
    # Save for running EpiGen inference
    df_cdr3 = mdata['gex'].obs['cdr3'].reset_index()
    df_cdr3 = df_cdr3.dropna(subset=['cdr3'])  # Exclude all NaN cases
    df_cdr3['label'] = 'AAAAA'
    df_cdr3 = df_cdr3.drop_duplicates(subset=['cdr3'], keep='first')
    df_cdr3.rename(columns={'cdr3': 'text'}, inplace=True)
    df_cdr3[['text', 'label']].to_csv(f"{outdir}/wu_formatted.csv", index=False)
    # Save for running EpiGen inference
    if outdir is None:
        outdir = "gex_obs"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    print(f"{outdir}/wu_formatted.csv was saved. ")
    return df_cdr3


def insert_epitope_info(mdata, pred_csv, outdir=None):
    # Insert the EpiGen prediction to mdata
    df = pd.read_csv(pred_csv)
    obs_df = mdata['gex'].obs

    # Ensure the cdr3 column in mdata['gex'].obs and tcr column in df are strings (if not already)
    obs_df['cdr3'] = obs_df['cdr3'].astype(str)
    df['tcr'] = df['tcr'].astype(str)

    # Reset the index to preserve 'cell_id'
    obs_df = obs_df.reset_index()
    obs_df = obs_df.rename(columns={'index': 'cell_id'})

    # Merge the predictions into the obs DataFrame based on matching cdr3 and tcr
    merged_df = obs_df.merge(df, how='left', left_on='cdr3', right_on='tcr')

    # Drop the 'tcr' column from the merged DataFrame, as it's now redundant
    merged_df.drop(columns=['tcr'], inplace=True)

    # Set the 'cell_id' column back as the index
    merged_df.set_index('cell_id', inplace=True)

    # Assign the merged DataFrame back to mdata['gex'].obs
    mdata['gex'].obs = merged_df

    # Save the result
    if outdir is None:
        outdir = 'gex_obs'
    Path(outdir).mkdir(parents=True, exist_ok=True)
    mdata['gex'].obs.to_csv(f"{outdir}/epitopes_added.csv", index=True)
    print(f"{outdir}/epitopes_added.csv")
    return mdata

def find_matches(pred_epitope, epi_db, threshold=1, method='levenshtein'):
    """
    Function to find matches based on the specified method.
    If a match is found, return 1, the ref_epitope, and ref_protein, else 0 and None values.
    """
    if not isinstance(pred_epitope, str):
        return 0, None, None

    if method == 'levenshtein':
        for ref_epitope, ref_protein in zip(epi_db['peptide'], epi_db['protein']):
            if Levenshtein.distance(pred_epitope, ref_epitope) <= threshold:
                return 1, ref_epitope, ref_protein
    elif method == 'substring':
        for ref_epitope, ref_protein in zip(epi_db['peptide'], epi_db['protein']):
            if pred_epitope in ref_epitope:
                return 1, ref_epitope, ref_protein

    return 0, None, None

def process_predictions(pred_column, epi_db, threshold=1, method='levenshtein'):
    """
    Function to process each prediction column in parallel.
    """
    with Pool(cpu_count()) as pool:
        results = pool.starmap(find_matches, [(epitope, epi_db, threshold, method) for epitope in pred_column])
    return results

def annotate_tumor_associated_epitopes(mdata, epi_db_path, method='levenshtein', threshold=1, topk=1, outdir=None):
    """
    Annotate cells in mdata for tumor association by querying epitope databases.
    Parameters:
    - mdata: The MuData object containing the cell data.
    - epi_db_path: Path to the CSV file containing the reference epitopes.
    - method: The matching method to use, either 'levenshtein' or 'substring'.
    - threshold: The maximum Levenshtein distance for a match (only used for 'levenshtein' method).
    """
    # Uniformize sequence lengths in pred_0 to pred_9 to a maximum of 9 characters
    for i in range(topk):  # Adjust the range as needed
        pred_column_name = f'pred_{i}'
        mdata['gex'].obs[pred_column_name] = mdata['gex'].obs[pred_column_name].apply(lambda x: x[:9] if isinstance(x, str) else x)

    # Load the epitope database
    epi_db = pd.read_csv(epi_db_path)

    for i in range(topk):  # Adjust the range as needed
        print(f"Start querying pred_{i}")
        pred_column_name = f'pred_{i}'
        match_column_name = f'match_{i}'
        ref_epitope_column_name = f'ref_epitope_{i}'
        ref_protein_column_name = f'ref_protein_{i}'

        # Run the matching process
        results = process_predictions(mdata['gex'].obs[pred_column_name], epi_db, threshold, method)

        # Unpack the results into separate columns
        mdata['gex'].obs[match_column_name], mdata['gex'].obs[ref_epitope_column_name], mdata['gex'].obs[ref_protein_column_name] = zip(*results)

    if outdir is None:
        outdir = "gex_obs"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    mdata['gex'].obs.to_csv(f"{outdir}/PA_marked.csv", index=False)
    print(f"{outdir}/PA_marked.csv was saved. ")
    return mdata

def construct_epitope_db(IEDB_db_path, TCIA_db_path, outdir):
    # Construct an epitope db file from TCIA and IEDB
    try:
        db_IEDB = pd.read_csv(IEDB_db_path)
        db_IEDB = db_IEDB.rename(columns={'Epitope - Name': 'peptide', 'Epitope - Source Molecule': 'protein'})
    except FileNotFoundError:
        raise FileNotFoundError(f"File {IEDB_db_path} not found.")
    except KeyError as e:
        raise KeyError(f"Missing column in IEDB data: {e}")

    # Read and process TCIA data
    try:
        db_tcia = pd.read_csv(TCIA_db_path, delimiter='\t')
        db_tcia = db_tcia.rename(columns={'gene': 'protein'})
    except FileNotFoundError:
        raise FileNotFoundError(f"File {TCIA_db_path} not found.")
    except KeyError as e:
        raise KeyError(f"Missing column in TCIA data: {e}")

    # Concatenate the DataFrames and return
    df = pd.concat([db_IEDB[['peptide', 'protein']], db_tcia[['peptide', 'protein']]], ignore_index=True)
    df.to_csv(f"{outdir}/tumor_associated_epitopes.csv", index=False)
    print(f"{outdir}/tumor_associated_epitopes.csv was saved. ")
    return df


def annotate_sites(mdata, col='cdr3_nt', outdir=None):
    # Assumes cdr3 already exists
    df = mdata['gex'].obs

    # Extract unique patients
    patients = df['patient'].unique()

    # Initialize an empty column for pattern
    df['pattern'] = ''

    for patient in patients:
        # Filter the DataFrame for the current patient
        patient_df = df[df['patient'] == patient]

        # Extract unique CDR3 sequences for the current patient
        cdr3s = set(patient_df[col].tolist())

        # Dictionary to hold the label for each cdr3 sequence
        cdr3_label_dict = {}

        for cdr3 in cdr3s:
            label = ''

            # Count occurrences in each site
            cdr3_in_tissue_cnt = patient_df[(patient_df[col] == cdr3) & (patient_df['source'] == 'Tumor')].shape[0]
            cdr3_in_NAT_cnt = patient_df[(patient_df[col] == cdr3) & (patient_df['source'] == 'NAT')].shape[0]
            cdr3_in_blood_cnt = patient_df[(patient_df[col] == cdr3) & (patient_df['source'] == 'Blood')].shape[0]

            # Tissue labeling
            if cdr3_in_tissue_cnt > 1:
                label += 'T'
            elif cdr3_in_tissue_cnt == 1:
                label += 't'
            else:
                label += 'x'

            # NAT labeling
            if cdr3_in_NAT_cnt > 1:
                label += 'N'
            elif cdr3_in_NAT_cnt == 1:
                label += 'n'
            else:
                label += 'x'

            # Blood labeling
            if cdr3_in_blood_cnt > 1:
                label += 'B'
            elif cdr3_in_blood_cnt == 1:
                label += 'b'
            else:
                label += 'x'

            # Store the label in the dictionary
            cdr3_label_dict[cdr3] = label

        # Map the labels back to the original DataFrame
        df.loc[df['patient'] == patient, 'pattern'] = df[col].map(cdr3_label_dict)

    # Assign the updated DataFrame back to mdata['gex'].obs
    mdata['gex'].obs = df

    if not outdir:
        outdir = "gex_obs"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    mdata['gex'].obs.to_csv(f"{outdir}/site_added.csv", index=False)
    print(f"{outdir}/site_added.csv")
    return mdata
