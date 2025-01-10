# EpiGen

EpiGen is a conditional generation model for predicting cognate epitope sequences using TCR sequence information. It aims to functionally annotate TCRs within TCR repertoires obtained from single-cell TCR sequencing (scTCR-seq).

## Overview
While previous analyses of TCR repertoires have focused on clonality expansion, the actual antigens each TCR targets remained elusive. EpiGen helps understand the existential role of each TCR by generating corresponding epitope sequences, which can then be queried to protein databases to check the underlying antigen information.

For more detailed information, please refer to our manuscript:
- "Generating cognate epitope sequences of T-cell receptors with a generative transformer (2024)"

## Resources
- Training Data: https://zenodo.org/records/14286754
- Model Checkpoints: https://zenodo.org/records/14286943

## Dependencies
The project relies on several core dependencies:
- `datasets`
- `huggingface`
- `transformer`
- `accelerate`
- `torch`
- `pandas`
- `numpy`
- `pickle`

For the comprehensive list of requirements, please refer to the `requirements.txt` file.

## Usage Guide

### Data Preparation
1. Create a CSV file containing TCR sequences with the following structure:
   - Required column: `text` (contains TCR sequences)
   - Required column: `label` (in current version, fill with placeholder text, e.g., 'ZZZZZ')

Example CSV format:
```csv
text,label
CASIPEGGRETQYF,ZZZZZ
CAVRATGTASKLTF,ZZZZZ
CASSGGNTPLVF,ZZZZZ
CASTRADTGELFF,ZZZZZ
CASEDSSDGANYGYTF,ZZZZZ
CASSELGARVYEQYF,ZZZZZ
```

### Model Setup
1. Download the pre-trained checkpoint:
   - Source: https://zenodo.org/records/14286943

2. Configure the inference script:
   - Open `scripts/train_eval_infer_EpiGen.sh`
   - Comment out training-related sections
   - Uncomment inference-related sections

### Running Inference
Use the following script configuration for inference:

```bash
datasets=("mydata")
for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python epigen/run_clm_predict.py \
        --model_name_or_path gpt2-small \
        --train_file data/train.csv \
        --validation_file data/${dataset}.csv \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --num_train_epochs 101 \
        --tokenizer_name regaler/EpiGen \
        --checkpointing_steps epoch \
        --with_tracking \
        --report_to wandb \
        --gradient_accumulation_steps 1 \
        --output_dir 241205_example_run \
        --inf_out_dir "241205_example_run/${dataset}" \
        --resume_from_checkpoint 241205_example_run/epoch_28 \
        --inference_mode \
        --gpt2_small
done
```

Key parameters to modify:
- `--validation_file`: Path to your input CSV file
- `--inf_out_dir`: Directory where inference results will be saved
- `--output_dir`: Root directory containing the model checkpoint
- `--resume_from_checkpoint`: Path to the downloaded checkpoint directory

The script will generate a CSV file in the specified `--inf_out_dir` with additional columns `pred_{i}` containing generated epitopes for each TCR.

### Epitope Database Query
To analyze the generated epitopes against known databases:

1. Download reference databases from https://zenodo.org/records/14344830:
   - `covid19_associated_epitopes.csv`: Compiled from IEDB
   - `tumor_associated_epitopes.csv`: Compiled from IEDB and TCIA

2. Available annotation functions:
   - For tumor-associated epitopes: Use `annotate_tumor_associated_epitopes()` in `cancer_wu/annotate.py`
   - For COVID-19-associated epitopes: Use `annotate_covid19_associated_epitopes()` in `covid19_su/annotate.py`

### Custom Analysis
You can create your own disease-specific epitope database and detection pipeline:
1. Construct a database of disease-specific epitopes
2. Adapt the annotation functions to detect disease-associated epitopes
3. Use the results to identify disease-associated T cells
4. Perform differential analyses between disease-associated and background T cells

Note: A more flexible API for epitope annotation will be supported in future versions.

## Quick Training Guide
### Applying Antigen Category Filter
See `construct_balanced_data()` in `test_antigen_category_filter()` in `EpiGen/scripts/run.py`. This function takes as input an intermediate dataset (such as `intermediate_data_train.csv`) and outputs a smaller, filtered dataset. You can change the seed to get the new filtered train, val, and test datasets. Then, manually add the public training set (`train.csv`) at the end of the filtered pseudo-labeled training set. For completeness, repeat this process to get the filtered validation and test sets. You may repeat this process three times to train three independent models to take an ensemble later. 

### Training a model
Use `EpiGen/scripts/train_eval_infer.sh` to train a new model with the newly filtered training set. After training, use the same script to run evaluation over epochs to choose the best model checkpoints. Use `val.csv` to select the best epoch. 

## Cancer dataset analysis
See `analysis/cancer_wu`

## COVID-19 dataset analysis
See `analysis/covid19_su`
