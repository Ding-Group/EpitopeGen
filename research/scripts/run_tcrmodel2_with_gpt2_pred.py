# This script runs tcrmodel2 sequentially using the converted pred file.
# 1. Run inference mode of EpiGen
# 2. Use convert_pred_to_tcrmodel2_format() of utils.py to convert the format
# 3. Use this script. Use slutm to run this script in parallel.
import pickle
import subprocess
import os

RESEARCH_PATH = ""
INPUT = "../coreset/data/tcrmodel2_args.pkl"

# Load the arguments from the pickle file
with open(INPUT, 'rb') as f:
    tcrmodel2_args_list = pickle.load(f)

# Set paths for databases and singularity SIF
alphafold_db = "RESEARCH_PATH/data/alphafold"
alphafold_sif = "RESEARCH_PATH/tcrmodel2/tcrmodel2.sif"
base_output_dir = "RESEARCH_PATH/tcrmodel2/outputs"

# Iterate through each argument set and execute the command
for args in tcrmodel2_args_list:
    output_dir = os.path.join(base_output_dir, args['tcr'])
    job_id = f"{args['tcr']}_{args['mhc']}_{args['epitope']}"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(f"{output_dir}/{job_id}"):
        continue
    else:
        # Construct the command
        command = [
            'singularity', 'run', '--nv', '-B', alphafold_db, alphafold_sif,
            '--job_id', job_id,
            '--output_dir', output_dir,
            '--tcra_seq', args['tcra_seq'],
            '--tcrb_seq', args['tcrb_seq'],
            '--pep_seq', args['pep_seq'],
            '--mhca_seq', args['mhca_seq'],
            '--ori_db', alphafold_db,
            '--tp_db', '/opt/tcrmodel2/data/databases',
            '--relax_structures', 'True'
        ]

        # Execute the command
        subprocess.run(command, check=True)

        print(f"Completed: {job_id}")

