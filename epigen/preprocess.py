import os
import math
import random
import subprocess
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import mhcgnomes

# local codes
from epigen.utils import is_valid_cdr, is_valid_peptide


def preprocess_netmhcpan(root, outdir="data", th=0.426, use_mhc=False):
    """
    Preprocess the NetMHCpan dataset.
    Select only positive binders from both BA and MS data if use_mhc is True.
    If use_mhc is False, consider only unique peptides and save.
    """
    amino_acids = set('ACDEFGHIKLMNPQRSTVWY')

    def is_valid_peptide(peptide):
        return 8 <= len(peptide) <= 12 and all(c in amino_acids for c in peptide)

    def load_and_filter_data(file_list, threshold):
        dfs = []
        for data in file_list:
            df = pd.read_csv(f"{root}/{data}", header=None, names=['peptide', 'label', 'allele_name'], sep=' ')
            df = df[df['label'] > threshold]
            dfs.append(df)
        return pd.concat(dfs)

    def process_with_mhc(data_ba, data_el):
        mhc_pseudo = pd.read_csv(f"{root}/MHC_pseudo.dat", header=None, names=['allele_name', 'allele'], sep=' ')
        allele_list = pd.read_csv(f"{root}/allelelist", sep='\s+', engine='python', header=None, names=['allele_name', 'HLA_Types'])

        df_ba = pd.concat([pd.merge(load_and_filter_data([data], th), mhc_pseudo, on='allele_name') for data in data_ba])

        df_el_list, df_filtered_out_list = [], []
        for data in data_el:
            df = load_and_filter_data([data], th)
            df_merged = pd.merge(df, mhc_pseudo, on='allele_name', how='left', indicator=True)
            df_el_list.append(df_merged[df_merged['_merge'] == 'both'])
            df_filtered_out_list.append(df_merged[df_merged['_merge'] == 'left_only'])

        df_el = pd.concat(df_el_list)
        df_filtered_out = pd.concat(df_filtered_out_list)
        df_filtered_out = pd.merge(df_filtered_out, allele_list, on='allele_name', how='left')
        df_filtered_out[['peptide', 'HLA_Types']].to_csv(f"{outdir}/peptides_multialleic.csv", index=False)

        df_ba[['peptide', 'allele']].to_csv(f"{outdir}/netmhcpan_ba.csv", index=False)
        df_el[['peptide', 'allele']].to_csv(f"{outdir}/netmhcpan_el.csv", index=False)
        pd.concat([df_ba, df_el])[['peptide', 'allele']].to_csv(f"{outdir}/netmhcpan.csv", index=False)

    def process_without_mhc(data_ba, data_el):
        peptides = set()
        for data in data_ba + data_el:
            df = pd.read_csv(f"{root}/{data}", header=None, names=['peptide', 'label', 'allele_name'], sep=' ')
            peptides.update([pep for pep in df['peptide'].unique() if is_valid_peptide(pep)])

        pd.DataFrame(list(peptides), columns=['peptide']).to_csv(f"{outdir}/netmhcpan_unique_peptides.csv", index=False)

    data_ba = [x for x in os.listdir(root) if x.endswith("ba")]
    data_el = [x for x in os.listdir(root) if x.endswith("el")]

    if use_mhc:
        process_with_mhc(data_ba, data_el)
    else:
        process_without_mhc(data_ba, data_el)

    print(f"NetMHCpan preprocessed data were saved under: {outdir}")


def _parse(stdout):
    split = stdout.split()
    ranks = []
    alleles = []
    for i in range(len(split)):
        if split[i] == 'PEPLIST':
            rank_el = float(split[i + 2])
            ranks.append(rank_el)
        elif split[i] == 'PEPLIST.':  # dot
            allele = split[i + 2]
            alleles.append(allele[:-1])
    try:
        best_allele = alleles[ranks.index(min(ranks))]
    except:
        best_allele = stdout
    return best_allele


def run_netmhcpan(pep_file, alleles):
    key = str(Path(pep_file).stem)
    cmd = [netmhcpan, '-p', pep_file, '-a', alleles]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode == 0:
        # Assuming _parse() is a function that you've defined elsewhere
        hla = _parse(result.stdout)
        return (key, hla)
    else:
        print(f"Error in running netMHCpan for {pep_file}")
        print(result.stderr)
        return (key, None)


def preprocess_netmhcpan_pred(netmhcpan, peptide_hla_csv, desc, batch=1000, outdir="data"):
    # Convert multi-allelic data into single-allelic by using NetMHCpan
    Path(f"{outdir}/__pycache__").mkdir(parents=True, exist_ok=True)
    Path(f"{outdir}/__pycache__/peptides_{desc}").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(peptide_hla_csv)
    # Save a batch of peptide files as csv
    cnt = 0
    alleles_list = []
    pep_hla = []
    pep_files = []
    while cnt < len(df):
        print(f"{cnt / len(df) * 100} \%")
        for i in range(cnt, min(cnt + batch, len(df))):
            if i == len(df):
                break
            row = df.iloc[i]
            pep = row['peptide']
            with open(f"{outdir}/__pycache__/peptides_{desc}/{pep}.csv", "w") as f:
                f.write(pep)

            alleles_list.append(row['HLA_Types'])
            pep_files.append(f"{outdir}/__pycache__/peptides_{desc}/{pep}.csv")
        # Create a pool of processes. Number of processes is by default the number of CPUs on the machine.
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(run_netmhcpan, pep_files, alleles_list))
        # Collect results
        pep_hla += [result for result in results if result[1] is not None]
        # Clear the cached csv files
        for pep in os.listdir(f"{outdir}/__pycache__/peptides_{desc}"):
            os.remove(f"{outdir}/__pycache__/peptides_{desc}/{pep}")
        alleles_list = []
        pep_files = []
        cnt += batch
    pep_hla = pd.DataFrame(pep_hla, columns=['peptide', 'allele'])
    pep_hla.to_csv(f"{outdir}/netmhcpan_multi2single_{desc}.csv", index=False)
    print(f"{outdir}/netmhcpan_multi2single_{desc}.csv was saved. ")


def preprocess_mhcflurry(root, outdir, use_mhc=False):
    """
    Preprocess the MHCflurry dataset.
    Get the positive binders of peptide and allele from MHCflurry if use_mhc is True.
    If use_mhc is False, gather unique peptides and save.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    amino_acids = set('ACDEFGHIKLMNPQRSTVWY')

    def is_valid_peptide(peptide):
        return 8 <= len(peptide) <= 12 and all(c in amino_acids for c in peptide)

    def process_with_mhc():
        files_info = {
            "Data_S1.csv": ['peptide', 'mhcflurry2.ba_best_allele'],
            "Data_S2.csv": ['peptide', 'hla'],
            "Data_S3.csv": ['peptide', 'allele'],
            "Data_S4.csv": ['peptide', 'allele'],
            "Data_S5.csv": ['peptide', 'hla']
        }

        for file, cols in files_info.items():
            df = pd.read_csv(f"{root}/{file}")
            if 'hit' in df.columns:
                df = df[df['hit'] == 1][cols]
            elif 'measurement_value' in df.columns:
                df['hit'] = df['measurement_value'].apply(lambda x: 1 - math.log(max(x, 0.1), 50000))
                df = df[df['hit'] > 0.42][cols]
            df.to_csv(f"{outdir}/{file.split('.')[0]}_processed.csv", index=False)
            del df

    def process_without_mhc():
        peptides = set()
        for file in os.listdir(root):
            if file.startswith("Data_S") and file.endswith(".csv"):
                df = pd.read_csv(f"{root}/{file}")
                peptides.update([pep for pep in df['peptide'].unique() if is_valid_peptide(pep)])

        df = pd.DataFrame(list(peptides), columns=['peptide'])
        df.to_csv(f"{outdir}/mhcflurry_unique_peptides.csv", index=False)

    if use_mhc:
        process_with_mhc()
    else:
        process_without_mhc()

    print(f"MHCflurry preprocessed data were saved under: {outdir}")


def preprocess_systeMHC(root, outdir, use_mhc=True):
    """
    Preprocess the systeMHC dataset.
    If use_mhc is True, gather peptide and allele information.
    If use_mhc is False, gather unique peptides and save.
    Only consider peptides with length between 8 and 12 and valid amino acid characters.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    files = [x for x in os.listdir(root) if x.endswith(".pepidx")]
    peptides = set()
    all_data = []
    amino_acids = set('ACDEFGHIKLMNPQRSTVWY')

    def is_valid_peptide(peptide):
        return 8 <= len(peptide) <= 12 and all(c in amino_acids for c in peptide)

    for file_ in tqdm(files):
        with open(f"{root}/{file_}", "r") as f:
            data = f.readlines()
        idx = data.index("### ===\n")
        data = data[idx+1:]
        valid_peptides = [x.split()[0] for x in data if is_valid_peptide(x.split()[0])]
        peptides.update(valid_peptides)
        if use_mhc:
            name = os.path.splitext(file_)[0].replace("_", ":")
            all_data += [(peptide, name) for peptide in valid_peptides]

    if use_mhc:
        df = pd.DataFrame(all_data, columns=['peptide', 'allele'])
        output_file = f"{outdir}/systeMHC_allele_specific.csv"
    else:
        df = pd.DataFrame(list(peptides), columns=['peptide'])
        output_file = f"{outdir}/systeMHC_unique_peptides.csv"

    df.to_csv(output_file, index=False)
    print(f"{output_file} was saved.")


def merge_preprocessed_data(data_root, MHC_pseudo, outname, use_mhc=False):
    # Function to standardize BoLA and Mamu
    def update_allele_format(allele, keyword="BoLA"):
        if allele.startswith(keyword):
            parts = allele.split('*')  # Split by '*'
            if len(parts) > 1:
                subparts = parts[1].split(':')  # Split the second part by ':'
                if len(subparts) > 1:
                    number = subparts[0]  # Get the numeric part
                    # Check if number length is 3 and starts with '0'
                    if len(number) == 3 and number.startswith('0'):
                        # Remove the leading '0'
                        new_number = number[1:]
                        # Recombine the allele string
                        return f"{parts[0]}*{new_number}:{subparts[1]}"
        return allele

    # Function to safely parse alleles with error handling
    def safe_parse_allele(allele):
        try:
            return mhcgnomes.parse(allele).to_string()
        except Exception as e:
            return None  # Return None for unparsable alleles

    def sanity_checks(df):
        # Check for NaN values and remove any rows with NaN
        initial_count = len(df)
        df = df.dropna()
        if len(df) < initial_count:
            print(f"Removed {initial_count - len(df)} rows containing NaN values.")

        # Remove duplicate rows
        initial_count = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_count:
            print(f"Removed {initial_count - len(df)} duplicate rows.")

        # Define a set of 20 standard amino acids
        amino_acids = set('ACDEFGHIKLMNPQRSTVWY')

        # Function to check if all characters in a string are amino acids
        def is_valid_sequence(x):
            return set(x).issubset(amino_acids)

        # Filter 'peptide' and 'mhc' columns for valid amino acid sequences
        valid_peptide = df['peptide'].apply(is_valid_sequence)
        valid_mhc = df['mhc'].apply(is_valid_sequence)
        df = df[valid_peptide & valid_mhc]
        if len(df) < initial_count:
            print(f"Filtered out invalid amino acid sequences, {initial_count - len(df)} rows removed.")

        print("Sanity check completed. Data is clean.")
        return df

    if not use_mhc:
        netmhcpan_file = f"{data_root}/netmhcpan/netmhcpan_unique_peptides.csv"
        mhcflurry_file = f"{data_root}/mhcflurry/mhcflurry_unique_peptides.csv"
        system_file = f"{data_root}/SysteMHC/systeMHC_unique_peptides.csv"

        # Read all and merge
        df_netmhcpan = pd.read_csv(netmhcpan_file)
        df_mhcflurry = pd.read_csv(mhcflurry_file)
        df_system = pd.read_csv(system_file)

        # Concatenate the dataframes
        df = pd.concat([df_netmhcpan, df_mhcflurry, df_system]).drop_duplicates().reset_index(drop=True)

        # Save the unique peptides
        df.to_csv(outname, index=False)
        exit(0)

    # Read and concatenate MHCflurry files
    mhcflurry_files = [f for f in os.listdir(f"{data_root}/mhcflurry") if f.startswith('S') and f.endswith('.csv')]
    df_mhcflurry = pd.concat([pd.read_csv(f"{data_root}/mhcflurry/{file}") for file in mhcflurry_files])

    # Read and concatenate NetMHCpan files from multi directory
    netmhcpan_files = os.listdir(f"{data_root}/netmhcpan/multi")
    df_net_multi = pd.concat([pd.read_csv(f"{data_root}/netmhcpan/multi/{file}") for file in netmhcpan_files])

    # Read SysteMHC data
    df_sys = pd.read_csv(f"{data_root}/SysteMHC/systeMHC_allele_specific.csv")

    # Combine all data frames into one and remove duplicates
    df_combined = pd.concat([df_mhcflurry, df_net_multi, df_sys]).drop_duplicates().reset_index(drop=True)

    # Reading MHC pseudo sequences from file
    with open(MHC_pseudo, "r") as file:
        mhc_pseudo = [line.strip().split() for line in file if line.strip()]
    df_pseudo = pd.DataFrame(mhc_pseudo, columns=['allele', 'mhc'])

    # Apply the safe_parse_allele function and drop rows with None (unparsable)
    df_combined['allele'] = df_combined['allele'].apply(safe_parse_allele)
    df_combined.dropna(subset=['allele'], inplace=True)
    df_pseudo['allele'] = df_pseudo['allele'].apply(safe_parse_allele)
    df_pseudo.dropna(subset=['allele'], inplace=True)

    # Apply update_allele_format for further standardization on allele names
    df_combined['allele'] = df_combined['allele'].apply(lambda x: update_allele_format(x, keyword="BoLA"))  # BoLA-2*008:01 --> BoLA-2*08:01
    df_pseudo['allele'] = df_pseudo['allele'].apply(lambda x: update_allele_format(x, keyword="BoLA"))  # BoLA-2*008:01 --> BoLA-2*08:01
    df_combined['allele'] = df_combined['allele'].apply(lambda x: update_allele_format(x, keyword="Mamu"))
    df_pseudo['allele'] = df_pseudo['allele'].apply(lambda x: update_allele_format(x, keyword="Mamu"))
    df_pseudo = df_pseudo.drop_duplicates().reset_index(drop=True)

    # Merge with MHC pseudo sequence data
    df_combined = pd.merge(df_combined, df_pseudo, on='allele', how='left')
    df_combined = df_combined[['peptide', 'mhc']]

    # Reading the netmhcpan.csv that already contains pseudo sequences
    df_net1 = pd.read_csv(f"{data_root}/netmhcpan/netmhcpan.csv")

    # Concatenate the combined data with df_net1 and remove any duplicates
    df_result = pd.concat([df_combined, df_net1]).drop_duplicates().reset_index(drop=True)

    # Final sanity check
    df_result = sanity_checks(df_result)

    # Saving the final result to CSV
    df_result.to_csv(f"{data_root}/{outname}", index=False)
    print(f"{data_root}/{outname} was saved. ")
    return df_result


def _save_filtered_df(filtered_df, outdir, outfile_name):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    outfile = f"{outdir}/{outfile_name}"
    filtered_df.to_csv(outfile, index=False)
    print(f"{outfile} was saved. ")

def _print_stat(filtered_df):
    print(filtered_df['tcr'].value_counts())
    print(filtered_df['epitope'].value_counts())
    print(f" *** Total rows: {len(filtered_df)}")

amino_acids = set('ACDEFGHIKLMNPQRSTVWY')


def preprocess_IEDB(data_path, outdir, min_epitope_length=8, max_epitope_length=12, min_original_seq_length=10, max_original_seq_length=16, min_cdr3b_length=10, max_cdr3b_length=20):
    """
    Filter the standardized IEDB file, change column names to construct the dataset
    to train binding affinity predictors
    """
    # Read the CSV data into a DataFrame
    df = pd.read_csv(data_path)

    # Filter the DataFrame based on the specified criteria
    filtered_df = df[
        (df['label'] == 1) &
        (df['epitopes'].apply(lambda x: min_epitope_length <= len(x) <= max_epitope_length)) &
        (df['original_seq'].apply(lambda x: min_original_seq_length <= len(x) <= max_original_seq_length)) &
        (df['trimmed_seq'].apply(lambda x: min_cdr3b_length <= len(x) <= max_cdr3b_length)) &
        (df['original_seq'].apply(lambda x: is_valid_cdr(x))) &
        (df['epitopes'].apply(lambda x: is_valid_peptide(x)))
    ]
    filtered_df = filtered_df.rename(columns={"original_seq": "tcr", "epitopes": "epitope"})
    # Save the result
    _save_filtered_df(filtered_df[['tcr', 'epitope']], outdir, "processed_IEDB.csv")
    _print_stat(filtered_df)
    return filtered_df


def preprocess_VDJdb(data_path, outdir, human_only=False, min_epitope_length=8, max_epitope_length=12, min_original_seq_length=10, max_original_seq_length=16, min_cdr3b_length=10, max_cdr3b_length=20):
    """
    Filter the standardized VDJdb file, change column names to construct the dataset
    to train binding affinity predictors
    """
    df = pd.read_csv(data_path)
    if human_only:
        filtered_df = df[(df['species'] == 'HomoSapiens') & (df['vdjdb.score'] > 0) & (df['mhc.class'] == 'MHCI')]
        print(f"VDJdb: total {len(filtered_df)} entries were selected ('HomoSapiens' and vdjdb.score > 0)")
        outfile = "processed_VDJdb_human_only.csv"
    else:
        filtered_df = df[(df['vdjdb.score'] > 0) & (df['mhc.class'] == 'MHCI')]
        print(f"VDJdb: total {len(filtered_df)} entries were selected (vdjdb.score > 0)")
        outfile = "processed_VDJdb.csv"
    filtered_df = filtered_df[
        (df['label'] == 1) &
        (df['antigen.epitope'].apply(lambda x: min_epitope_length <= len(x) <= max_epitope_length)) &
        (df['cdr3'].apply(lambda x: min_original_seq_length <= len(x) <= max_original_seq_length)) &
        (df['cdr3'].apply(lambda x: is_valid_cdr(x))) &
        (df['antigen.epitope'].apply(lambda x: is_valid_peptide(x)))
    ]
    filtered_df = filtered_df.rename(columns={"cdr3": "tcr", "antigen.epitope": "epitope"})

    # Select the required columns and add the 'HLA' column from 'mhc.a'
    filtered_df['HLA'] = filtered_df['mhc.a']

    # Only consider human HLA
    filtered_df['HLA'] = filtered_df['HLA'].apply(lambda x: x if isinstance(x, str) and x.startswith("HLA") else None)

    # Drop rows where 'HLA' is None
    filtered_df = filtered_df.dropna(subset=['HLA'])

    # Handle multiple HLA entries by keeping only the first one
    filtered_df['HLA'] = filtered_df['HLA'].apply(lambda x: x.split(',')[0] if ',' in x else x)

    def standardize_hla_name(x):
        x = mhcgnomes.parse(x).to_string()
        x = x[4:]
        splits = x.split(":")
        if len(splits) == 1:
            return splits[0] + ":" + '01'
        elif len(splits) == 2:
            return splits[0] + ":" + splits[1]
        elif len(splits) ==3:
            return splits[0] + ":" + splits[1]

    # Standardize the HLA name to 4 digits
    filtered_df['HLA'] = filtered_df['HLA'].apply(lambda x: standardize_hla_name(x))

    # Save the result
    _save_filtered_df(filtered_df[["tcr", "epitope", "HLA"]], outdir, outfile)
    _print_stat(filtered_df)
    return filtered_df


def preprocess_PIRD(data_path, outdir, min_epitope_length=8, max_epitope_length=12, min_original_seq_length=10, max_original_seq_length=16, min_cdr3b_length=10, max_cdr3b_length=20):
    """
    Filter the standardized PIRD file, change column names to construct the dataset
    to train binding affinity predictors
    """
    df = pd.read_csv(data_path)
    df = df[df['Antigen.sequence'].apply(lambda x: len(x) > 1)]
    filtered_df = df[
        (df['label'] == 1) &
        (df['Antigen.sequence'].apply(lambda x: min_epitope_length <= len(x) <= max_epitope_length)) &
        (df['CDR3.beta.aa'].apply(lambda x: min_original_seq_length <= len(x) <= max_original_seq_length)) &
        (df['CDR3.beta.aa'].apply(lambda x: is_valid_cdr(x))) &
        (df['Antigen.sequence'].apply(lambda x: is_valid_peptide(x)))
    ]
    filtered_df = filtered_df.rename(columns={"CDR3.beta.aa": "tcr", "Antigen.sequence": "epitope"})
    # Save the result
    _save_filtered_df(filtered_df[["tcr", "epitope"]], outdir, "processed_PIRD.csv")
    _print_stat(filtered_df)
    return filtered_df


def preprocess_McPAS(data_path, outdir, min_epitope_length=8, max_epitope_length=12, min_original_seq_length=10, max_original_seq_length=16, min_cdr3b_length=10, max_cdr3b_length=20):
    """
    Filter the standardized McPAS file, change column names to construct the dataset
    to train binding affinity predictors
    """
    df = pd.read_csv(data_path)
    # Filter the DataFrame based on the specified criteria
    df = df[df['Epitope.peptide'].apply(lambda x: isinstance(x, str))]
    df = df[df['CDR3.beta.aa'].apply(lambda x: isinstance(x, str))]
    filtered_df = df[
        (df['label'] == 1) &
        (df['Epitope.peptide'].apply(lambda x: min_epitope_length <= len(x) <= max_epitope_length)) &
        (df['CDR3.beta.aa'].apply(lambda x: min_original_seq_length <= len(x) <= max_original_seq_length)) &
        (df['CDR3.beta.aa'].apply(lambda x: is_valid_cdr(x))) &
        (df['Epitope.peptide'].apply(lambda x: is_valid_peptide(x)))
    ]
    filtered_df = filtered_df.rename(columns={"CDR3.beta.aa": "tcr", "Epitope.peptide": "epitope"})
    # Save the result
    _save_filtered_df(filtered_df[["tcr", "epitope"]], outdir, "processed_McPAS.csv")
    _print_stat(filtered_df)
    return filtered_df


def preprocess_Glanville(data_path, outdir, min_epitope_length=8, max_epitope_length=12, min_original_seq_length=10, max_original_seq_length=16, min_cdr3b_length=10, max_cdr3b_length=20):
    def norm_hla_name(x):
        if x == 'HLA-A2':
            return "A*02:01"
        elif x == 'HLA-A1':
            return "A*01:01"
        elif x == 'HLA-B7':
            return "B*07:01"

    df = pd.read_csv(data_path)
    # Filter the DataFrame based on some criteria
    df = df[df['Antigen-peptide'].apply(lambda x: isinstance(x, str))]
    df = df[df['CDR3b'].apply(lambda x: isinstance(x, str))]
    filtered_df = df[
        (df['label'] == 1) &
        (df['CDR3b'].apply(lambda x: is_valid_cdr(x))) &
        (df['Antigen-peptide'].apply(lambda x: is_valid_peptide(x)))
    ]
    filtered_df = filtered_df.rename(columns={"CDR3b": "tcr", "Antigen-peptide": "epitope"})
    filtered_df['HLA'] = filtered_df['HLA'].apply(lambda x: norm_hla_name(x))
    # Save the result
    _save_filtered_df(filtered_df[["tcr", "epitope", "HLA"]], outdir, "processed_Glanville.csv")
    _print_stat(filtered_df)
    return filtered_df


def generate_shuffled_negatives_for_testset(test_sets, outdir, multiplier=1, use_hla=False, desc=None):
    """
    Generate negative samples for the test set.
    test_sets: List of test csv files for each subgroup.
    multiplier: Ratio between positive and negative samples.
    """
    all_dfs = []

    for testset in test_sets:
        # Read each test set
        df = pd.read_csv(testset)
        df['source'] = os.path.basename(testset).split("_")[1]
        df['label'] = 1

        # Generate negatives by shuffling. Avoid reassigning existing pairs
        if use_hla:
            positive_pairs = set(zip(df['tcr'], df['epitope'], df['HLA']))
            epitope_hlas = list(zip(df['epitope'], df['HLA']))
        else:
            positive_pairs = set(zip(df['tcr'], df['epitope']))
            epitope_hlas = list(set(df['epitope'].tolist()))
        tcrs = df['tcr'].tolist()
        num_negatives = len(df) * multiplier
        negative_pairs = set()

        if use_hla:
            while len(negative_pairs) < num_negatives:
                tcr = random.choice(tcrs)
                epitope, hla = random.choice(epitope_hlas)
                if (tcr, epitope, hla) not in positive_pairs and (tcr, epitope, hla) not in negative_pairs:
                    negative_pairs.add((tcr, epitope, hla))
        else:
            while len(negative_pairs) < num_negatives:
                tcr = random.choice(tcrs)
                epitope = random.choice(epitope_hlas)  # No HLA info
                if (tcr, epitope) not in positive_pairs and (tcr, epitope) not in negative_pairs:
                    negative_pairs.add((tcr, epitope))

        if use_hla:
            negative_df = pd.DataFrame(list(negative_pairs), columns=['tcr', 'epitope', 'HLA'])
        else:
            negative_df = pd.DataFrame(list(negative_pairs), columns=['tcr', 'epitope'])
        negative_df['source'] = df['source'].iloc[0]
        negative_df['label'] = 0

        # Concatenate positive and negative samples for the current test set
        combined_df = pd.concat([df, negative_df]).reset_index(drop=True)
        all_dfs.append(combined_df)

    final_df = pd.concat(all_dfs).reset_index(drop=True)

    # After the for-loop, before saving
    for source, group in final_df.groupby('source'):
        num_positive = group[group['label'] == 1].shape[0]
        num_negative = group[group['label'] == 0].shape[0]
        print(f"Source: {source} - Positive samples: {num_positive}, Negative samples: {num_negative}")

    # Concatenate all combined dataframes from each test set
    final_df = pd.concat(all_dfs).reset_index(drop=True)
    if desc:
        outfile = f"{outdir}/{desc}_test_with_neg_multi_{multiplier}.csv"
    else:
        outfile = f"{outdir}/test_with_neg_multi_{multiplier}.csv"

    final_df.to_csv(outfile, index=False)
    print(f"{outfile} was saved. ")

    return final_df


def generate_positive_only_testset_for_eval(test_sets, outdir, use_hla=False, desc=None):
    """
    Generate a test set for evaluation composed only of positive samples.

    Parameters:
    test_sets: List of test csv files for each subgroup.
    outdir: Directory to save the output file.
    use_hla: Boolean to indicate whether to consider the HLA column.
    """
    dfs = []
    for testset in test_sets:
        df = pd.read_csv(testset)
        # Add necessary columns
        df['source'] = os.path.basename(testset).split("_")[1]
        df['label'] = 1

        if use_hla and 'HLA' not in df.columns:
            raise ValueError(f"The file {testset} does not contain the HLA column, but use_hla is set to True.")

        dfs.append(df)

    df = pd.concat(dfs).reset_index(drop=True)

    # Include HLA column if use_hla is True
    if use_hla:
        columns_to_include = ['tcr', 'epitope', 'HLA', 'source', 'label']
    else:
        columns_to_include = ['tcr', 'epitope', 'source', 'label']

    if desc:
        outfile = f"{outdir}/{desc}_test_pos_only.csv"
    else:
        outfile = f"{outdir}/test_pos_only.csv"
    df[columns_to_include].to_csv(outfile, index=False)
    print(f"{outfile} was saved.")


def generate_externally_paired_negatives_for_testset(test_sets, external_tcr_set, outdir, multiplier=1, desc=None):
    """
    Generate negative samples for the test set using an external TCR set, considering HLA information.

    Parameters:
    test_sets: List of test csv files for each subgroup.
    external_tcr_set: Path to the external TCR set CSV file.
    outdir: Directory to save the output file.
    multiplier: Ratio between positive and negative samples.
    """
    dfs = []
    for testset in test_sets:
        df = pd.read_csv(testset)
        # Add necessary columns
        df['source'] = os.path.basename(testset).split("_")[1]
        df['label'] = 1
        dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)

    # Generate negatives by shuffling. Avoid reassigning existing pairs
    positive_pairs = set(zip(df['tcr'], df['epitope'], df['HLA']))
    df_ext = pd.read_csv(external_tcr_set)
    # Apply sanity check
    df_ext = df_ext[df_ext['tcr'].apply(is_valid_cdr)]

    tcrs = df_ext['tcr'].tolist()
    epitope_hlas = list(zip(df['epitope'], df['HLA']))
    num_negatives = len(df) * multiplier
    negative_pairs = set()

    while len(negative_pairs) < num_negatives:
        tcr = random.choice(tcrs)
        epitope, hla = random.choice(epitope_hlas)
        if (tcr, epitope, hla) not in positive_pairs and (tcr, epitope, hla) not in negative_pairs:
            negative_pairs.add((tcr, epitope, hla))

    negative_df = pd.DataFrame(list(negative_pairs), columns=['tcr', 'epitope', 'HLA'])
    negative_df['source'] = 'external'  # Assuming the source is the same for all negatives
    negative_df['label'] = 0

    # Concatenate positive and negative samples
    final_df = pd.concat([df, negative_df]).reset_index(drop=True)

    # Save the final combined DataFrame
    Path(outdir).mkdir(parents=True, exist_ok=True)
    if desc:
        outfile = os.path.join(outdir, f'{desc}_test_ext_paired_neg_multi_{multiplier}.csv')
    else:
        outfile = os.path.join(outdir, f'_test_ext_paired_neg_multi_{multiplier}.csv')
    final_df.to_csv(outfile, index=False)
    print(f"{outfile} was saved.")


def generate_ext_peptide_paired_neg_for_testset(test_sets, external_pep_set, outdir, multiplier=1, use_hla=False, desc=None):
    dfs = []
    for testset in test_sets:
        df = pd.read_csv(testset)
        # Add necessary columns
        df['source'] = os.path.basename(testset).split("_")[1]
        df['label'] = 1
        dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)

    # Generate negatives by shuffling. Avoid reassigning existing pairs
    if use_hla:
        positive_pairs = set(zip(df['tcr'], df['epitope'], df['HLA']))
    else:
        positive_pairs = set(zip(df['tcr'], df['epitope']))
    df_ext = pd.read_csv(external_pep_set)
    # Apply sanity check
    df_ext = df_ext[df_ext['peptide'].apply(is_valid_peptide)]

    peptides = df_ext['peptide'].tolist()  # external peptides
    if use_hla:
        tcr_hlas = list(zip(df['tcr'], df['HLA']))
    else:
        tcr_hlas = list(df['tcr'].unique())
    num_negatives = len(df) * multiplier
    negative_pairs = set()


    if use_hla:
        while len(negative_pairs) < num_negatives:
            pep = random.choice(peptides)
            tcr, hla = random.choice(tcr_hlas)
            if (tcr, pep, hla) not in positive_pairs and (tcr, pep, hla) not in negative_pairs:
                negative_pairs.add((tcr, pep, hla))
    else:
        while len(negative_pairs) < num_negatives:
            pep = random.choice(peptides)
            tcr = random.choice(tcr_hlas)
            if (tcr, pep) not in positive_pairs and (tcr, pep) not in negative_pairs:
                negative_pairs.add((tcr, pep))

    if use_hla:
        negative_df = pd.DataFrame(list(negative_pairs), columns=['tcr', 'epitope', 'HLA'])
    else:
        negative_df = pd.DataFrame(list(negative_pairs), columns=['tcr', 'epitope'])

    negative_df['source'] = 'external'  # Assuming the source is the same for all negatives
    negative_df['label'] = 0

    # Concatenate positive and negative samples
    final_df = pd.concat([df, negative_df]).reset_index(drop=True)

    # Save the final combined DataFrame
    Path(outdir).mkdir(parents=True, exist_ok=True)
    if desc:
        outfile = os.path.join(outdir, f'{desc}_test_ext_peptide_paired_neg_multi_{multiplier}.csv')
    else:
        outfile = os.path.join(outdir, f'test_ext_peptide_paired_neg_multi_{multiplier}.csv')
    final_df.to_csv(outfile, index=False)
    print(f"{outfile} was saved.")


def find_data_source(data, iedb, vdjdb, pird, mcpas):
    """
    Find which data source each row came from in train/val/test sets.
    Possible sources are: IEDB, VDJdb, PIRD, and McPAS.
    If no match is found, the source is set to 'vdjdb'.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added 'source' column indicating the data source.
    """

    # Read the data
    df = pd.read_csv(data)
    df_iedb = pd.read_csv(iedb)
    df_vdjdb = pd.read_csv(vdjdb)
    df_pird = pd.read_csv(pird)
    df_mcpas = pd.read_csv(mcpas)

    # Initialize the 'source' column with 'vdjdb'
    df['source'] = 'vdjdb'

    # Helper function to check and update source
    def update_source(row, source_df, source_name):
        match = source_df[(source_df['tcr'] == row['tcr']) & (source_df['epitope'] == row['epitope'])]
        if not match.empty:
            return source_name
        return row['source']

    # Iterate through each row and determine the source
    for index, row in df.iterrows():
        df.at[index, 'source'] = update_source(row, df_pird, 'pird')
        if df.at[index, 'source'] == 'pird':
            continue

        df.at[index, 'source'] = update_source(row, df_mcpas, 'mcpas')
        if df.at[index, 'source'] == 'mcpas':
            continue

        df.at[index, 'source'] = update_source(row, df_iedb, 'iedb')
        if df.at[index, 'source'] == 'iedb':
            continue

        df.at[index, 'source'] = update_source(row, df_vdjdb, 'vdjdb')

    df.to_csv("test.csv", index=False)
    print("test.csv was saved. ")
    return df


def sample_and_format_MIRA_data(data, sample, outdir, use_mhc=False, save_epitope=False):
    def is_valid_cdr(cdr):
        return 10 <= len(cdr) <= 20 and all(c in amino_acids for c in cdr)

    df = pd.read_csv(data)
    df['tcr'] = df['TCR BioIdentity'].apply(lambda x: x.split("+")[0])

    if use_mhc:
        MHCs = [
            "YFAMYGEKVAHTHVDTLYGVRYDHYYTWAVLAYTWYA",  # HLA-A*02:01
            "YYTKYREISTNTYENTAYGIRYDDDYTWAVDAYLSYV",  # HLA-B*44:02
            "YYTKYREISTNTYENTAYGIRYDDDYTWAVLAYLSYV"  # HLA-B*44:03
        ]
        # Create a DataFrame for the MHCs
        mhc_df = pd.DataFrame(MHCs, columns=['MHC'])

        # Repeat each row of df 3 times
        df_repeated = df.loc[df.index.repeat(3)].reset_index(drop=True)

        # Tile the MHCs to match the length of the repeated DataFrame
        mhc_repeated = pd.concat([mhc_df] * (len(df_repeated) // len(mhc_df) + 1)).reset_index(drop=True)

        # Combine the repeated CDR3s with the MHCs
        result_df = df_repeated.copy()
        result_df['MHC'] = mhc_repeated['MHC']

        result_df['text'] = result_df['tcr'] + "|" + result_df['MHC']
    else:
        result_df = df.copy()
        result_df['text'] = result_df['tcr']
        result_df = result_df[result_df['text'].apply(lambda x: is_valid_cdr(x))]

    result_df['label'] = "AAAAA"

    # Sample unique TCRs
    unique_tcrs = result_df['tcr'].drop_duplicates().sample(n=sample, random_state=42)
    epitopes_list = df.sample(n=sample, random_state=42)['Amino Acids'].tolist()
    epitopes_result = []
    for item in epitopes_list:
        epitopes_result += item.split(",")

    # Create a new dataframe with only the first occurrence of each unique TCR
    sampled_df = result_df[result_df['tcr'].isin(unique_tcrs)].drop_duplicates(subset='tcr')

    # Ensure we have exactly 'sample' number of rows
    if len(sampled_df) < sample:
        print(f"Warning: Only {len(sampled_df)} unique TCRs found, less than the requested {sample}")
    elif len(sampled_df) > sample:
        sampled_df = sampled_df.head(sample)

    # Generate the output file name
    name = Path(data).stem
    Path(outdir).mkdir(parents=True, exist_ok=True)
    outfile = os.path.join(outdir, f"{name}_sampled.csv")

    # Save the sampled DataFrame to a CSV file
    sampled_df[['text', 'label']].to_csv(outfile, index=False)
    print(f"{outfile} was saved with {len(sampled_df)} unique TCRs.")
