# Development Guide

This documentation outlines the complete development process for our TCR-epitope binding prediction system.

## Process Overview

The development process consists of the following key steps:

1. [Data Collection](#data-collection)
2. [Robust Affinity Predictor (RAP)](#robust-affinity-predictor)
3. [Feature Extraction](#feature-extraction)
4. [Label Propagation](#label-propagation)
5. [Redundancy Filter](#redundancy-filter)
6. [Antigen Category Filter](#antigen-category-filter)
7. [Training a Tokenizer](#training-a-tokenizer)
8. [Training a GPT-2 Model](#training-a-gpt2-model)
9. [Evaluating Binding Affinity](#evaluating-binding-affinity)
10. [Evaluation using InterfaceAnalyzer](#evaluation-using-interfaceanalyzer)
11. Cancer Dataset Analysis: See `analysis/cancer_wu`
12. COVID-19 Dataset Analysis: See `analysis/covid19_su`

## Data Collection

For detailed information, please refer to the Reporting Summary section of our manuscript. Our data collection encompasses multiple sources:

### Paired TCR-epitope Data
We integrated data from four established databases:
- IEDB (https://www.iedb.org)
- VDJdb (https://github.com/antigenomics/vdjdb-db)
- PIRD (https://github.com/wukevin/tcr-bert/blob/main/data/pird/pird_tcr_ab.csv)
- McPAS-TCR (https://friedmanlab.weizmann.ac.il/McPAS-TCR/)

### Unpaired Data
- **Epitope sequences** sourced from:
  - NetMHCpan-4.1 (https://services.healthtech.dtu.dk/suppl/immunology/NAR_NetMHCpan_NetMHCIIpan/)
  - MHCFlurry v2.0 (https://data.mendeley.com/datasets/zx3kjzc3yx/3)
  - SysteMHC (https://systemhc.sjtu.cn/)
- **TCR sequences** obtained from:
  - TCRdb (https://guolab.wchscu.cn/TCRdb/#/)

### Validation Data
For model validation, we utilized:
- Independent test sets from:
  - Glanville et al. (https://www.nature.com/articles/nature22976#additional-information)
  - Nolan et al. (https://clients.adaptivebiotech.com/pub/covid-2020)
- Additional real-world application datasets:
  - CD8+ T cell dataset (10X Genomics: https://www.10xgenomics.com/datasets/cd-8-plus-t-cells-of-healthy-donor-1-1-standard-3-0-2)
  - Cancer dataset (Gene Expression Omnibus: GSE139555)
  - COVID-19 dataset (ArrayExpress: E-MTAB-9357)

## Robust Affinity Predictor

Our label propagation system needs to analyze over 70 billion pairs with high reliability, recall, and precision. Initial testing revealed suboptimal binding affinity distribution in existing models at this scale. We enhanced the TABR-BERT model (https://github.com/Freshwind-Bioinformatics/TABR-BERT) with three key modifications:

1. **Cross-Entropy Loss Implementation**
   - Replaced ranking loss with `CrossEntropyLoss()`
   - Improved discrimination between positive and negative samples

2. **MHC Information Removal**
   - Eliminated MHC-related architectures due to limited public (TCR, epitope, MHC) triplet data
   - This decision improved epitope diversity, though future work should address this limitation

3. **Enhanced Negative Pair Generation**
   - Implemented three generation schemes:
     - Shuffling positive pairs (assuming random pairing yields low affinity)
     - Pairing TCRs with external epitopes
     - Pairing epitopes with external TCRs

### Implementation Details

Our implementation extends the TABR-BERT repository with new files:
- `pre_train_peptide_embedding_model.py`
- `train_tcr_peptide_prediction_model.py`
- `train_tcr_pmhc_prediction_model.py`
- `tcr_pmhc_model.py`
- `select_tcr_pep_model.py`

### Training Process

1. Train a peptide-only embedding model using `tabr_bert_fork/pre_train_peptide_embedding_model.py`
2. Utilize the TCR embedding model from TABR-BERT (`tabr_bert_fork/model/tcr_model.pt`)
3. Train binding affinity predictors:
   - For standard prediction: `train_tcr_peptide_prediction_model.py`
   - For MHC incorporation: `train_tcr_pmhc_prediction_model.py`
4. Train five independent models for ensemble prediction
5. Use `select_tcr_pep_model.py` to identify the best epoch for each model

All model files are available at: https://zenodo.org/records/14286943

## Feature Extraction

Before running label propagation, extract features from both TCR and epitope sequences.

### Epitope Feature Extraction

Use `EpitopeFeaturizer` from `epigen/featurize.py`:

```python
featurizer = EpitopeFeaturizer(
    epitope_data="data/240606_unique_peptides/candidate_peptides.csv",
    model_path="tabr_bert_fork/output/pmhc_models_peptide_only/pmhc_model_e100.pt",
    pseudo_sequence_file="tabr_bert_fork/data/mhcflurry.allele_sequences_homo.csv",
    outdir="data/epitope_features",
    use_mhc=False
)
featurizer.featurize_epitopes()
```

This generates `epitope_features/{epi}_features_{i}.pkl` containing dictionaries mapping peptides (or peptide_mhc) to numpy feature arrays.

### TCR Feature Extraction

Use `TCRDBFeaturizer` from `epigen/featurize.py`:

```python
featurizer = TCRDBFeaturizer(
    tcr_data="data/240612_tcrdb/tcrs_for_candidate.csv",
    model_path="tabr_bert_fork/model/tcr_model.pt",
    outdir="data/tcr_features"
)
featurizer.featurize_tcrs()
```

This creates `tcr_features_{i}.pkl` files in the specified output directory.

## Label Propagation

Use `epigen/label_prop.py` to identify high-affinity (TCR, epitope) pairs using the Robust Affinity Predictor:

```python
sampler = TCRPepSampler(
    tcr_feat_pkl="data/tcr_features/tcr_features_{i}.pkl",
    pep_feat_root="data/epitopes_features",
    model_paths=[
        "tabr_bert_fork/output/240612_rand_tcr_pmhc_1/tcr_pep_e58.pt",
        "tabr_bert_fork/output/240612_rand_tcr_pmhc_2/tcr_pep_e72.pt",
        "tabr_bert_fork/output/240612_rand_tcr_pmhc_3/tcr_pep_e60.pt",
        "tabr_bert_fork/output/240612_rand_tcr_pmhc_4/tcr_pep_e50.pt",
        "tabr_bert_fork/output/240612_rand_tcr_pmhc_5/tcr_pep_e50.pt",
    ],
    outdir="affinity_tables",
    tcr_chunk=4096,
    batch_size=16
)
sampler.sample()
```

### Processing Requirements

- Run this code for all TCR feature files (typically ~25)
- Parallelize jobs based on available computing resources
- Approximate processing time: 4 days using 10 NVIDIA L40S GPUs

### Output Structure

The process creates directories like:
```
affinity_tables_tcr_data_0_20240617_221652
affinity_tables_tcr_data_0_20240618_104331
affinity_tables_tcr_data_1_20240618_210752
...
affinity_tables_tcr_data_22_20240619_141631
```

Each directory contains pickle and CSV files:
```
sampled_data_4096.pkl
sampled_data_8192.pkl
...
```

Each pickle file contains data for 4096 TCRs (chunk size), with each TCR entry containing the 256 epitope sequences showing highest binding affinity.

### Post-processing

Use `postprocess_sampled_data()` from `epigen/utils.py` to format the data:

```python
postprocess_sampled_data(
    root="data_v6",
    keyword="affinity_tables",
    outdir="data/240930_topk32_100",
    topk=32,
    use_mhc=False
)
```

Note: The default `topk=32` means selecting the top 32 epitopes with highest binding affinity for each TCR. This can result in a TCR appearing up to 32 times in different rows of the final dataset, potentially generating around 50 million TCR-epitope pairs depending on parameters. 

## Redundancy Filter
To ensure data quality, apply the Redundancy Filter to the intermediate dataset to remove epitope sequences that appear more than `n_max_epi` times. This can be accomplished using the `remove_redundancy()` function from `epigen/utils.py`. You can adjust the threshold (`n_max_epi`) based on your specific requirements:

```
remove_redundancy("data/240930_topk32_100/all_data_topk32.csv", th=th)
```

This redundancy removal step is crucial when applying EpiGen to scTCR-seq datasets. Since we typically process approximately 100,000 T cells per analysis, maintaining appropriate distributional characteristics of the generated epitopes is essential.

## Antigen Category Filter
After applying the Redundancy Filter, implement the Antigen Category Filter to calibrate the distribution of antigen sources for epitopes. This filter enforces distributional constraints based on established immunological principles for CD8+ T cells:

1. Viral Dominance
2. Limited Bacteria
3. Endogenous Presence
4. Rare Fungi and Parasites
5. No Reported Pathogenic Archaea

To implement this filter, first identify the species associated with each epitope using BLASTP. Use the `run_blastp()` and `create_EpiGen_table()` functions from `epigen/antigen_category_filter.py`. Begin by identifying tumor-associated and self-antigens.

Next, establish the following relationships:
species info → NCBI accession numbers → taxonomy ID → species lineage → antigen category

Prerequisites:
- Install BLASTP from https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/
- Install Entrez Direct tools from https://www.ncbi.nlm.nih.gov/books/NBK179288/

You may optionally partition the table to improve processing speed:

```
merge_and_partition_epigen_tables(
    outdir="data/partitioned_epigen_tables",
    epigen_table_train=f"predictions/241106_all_data_topk32_th_100_train/tables/240620_all_data_topk32_th_100_train.csv",
    epigen_table_val=f"predictions/241106_all_data_topk32_th_100_val/tables/240620_all_data_topk32_th_100_val.csv",
    epigen_table_test=f"predictions/241106_all_data_topk32_th_100_test/tables/240620_all_data_topk32_th_100_test.csv",
    n_part=10
)

accessions_list_from_table(
    epigen_table=f"{root}/donor1_tcr_repertoire_trb_formatted.csv",
    tumor_csv=f"{root}/partitions/tumor.csv",
    self_csv=f"{root}/partitions/self.csv",
    col='pred_0',
    desc=desc
)

accession2taxid(
    accession_list=f"{root}/accessions_list_{desc}.txt",
    desc=desc,
    chunk_size=20000
)

run_efetch_parallel(
    tax_ids_file=f"{root}/result_{desc}.txt",
    output_dir=f"{root}/efetch",
    chunk_size=200
)

# Parse the lineage information
efetch_dir = f"{root}/efetch"
result = []
for file in tqdm(os.listdir(efetch_dir)):
    category = parse_efetch_result(xml_file=f"{efetch_dir}/{file}")
    tax_id = file[:-4].split("_")[1]
    result.append((tax_id, category))
df = pd.DataFrame(result, columns=['tax_id', 'category'])
df.to_csv(f"{root}/tax_id2category_{desc}.csv", index=False)

make_species2category(
    accessions_list=f"{root}/accessions_list_{desc}.txt",
    accession2tax_id_result=f"{root}/result_{desc}.txt",
    tax_id2category=f"{root}/tax_id2category_{desc}.csv",
    epigen_table=f"{root}/donor1_tcr_repertoire_trb_formatted.csv",
    outdir=root
)
```

Simultaneously, identify tumor-associated antigens and self-antigens using functions from `epigen/antigen_category_filter.py`. This requires downloading the database of tumor-associated epitopes from IEDB and TCIA:

```
identify_tumor_antigens(
    epigen_table=f"{root}/donor1_tcr_repertoire_trb_formatted.csv",
    epi_db_path="cancer_wu/data/tumor_associated_epitopes.csv",
    outdir=f"{root}/tumor_marked",
    col="pred_0",
    threshold=0,
    method='substring',
    debug=None
)

retrieve_rows_tumor_associated(
    epigen_table=f"{root}/donor1_tcr_repertoire_trb_formatted.csv",
    parted_tumor_antigen_annotation_root=f"{root}/tumor_marked",
    col="pred_0"
)

retrieve_rows_self_antigens(
    epigen_table=f"{root}/donor1_tcr_repertoire_trb_formatted.csv",
    tumor_csv=f"{root}/partitions/tumor.csv",
    col="pred_0"
)
```

With the antigen category information, construct the balanced dataset:

```
add_category_annotation(
    epigen_table=f"{root}/donor1_tcr_repertoire_trb_formatted.csv",
    species2category=f"{root}/species2category.csv",
    tumor_csv=f"{root}/partitions/tumor.csv",
    self_csv=f"{root}/partitions/self.csv",
    col='pred_0'
)

for split in ["train", "val", "test"]:
    construct_balanced_data(
        outdir=f"predictions/241106_all_data_topk32_th_100_{split}/tables",
        annotated_table=f"{root}/240620_all_data_topk32_th_100_{split}_cat_annotated.csv",
        seed=seed
    )
```

The final dataset used for training, validation, and testing of EpiGen is available at https://zenodo.org/records/14286754.

## Training a Tokenizer
Before training a GPT2 model, you must train a specialized tokenizer for TCR and epitope sequences. This process will generate a `vocab.json` file defining the vocabularies. Use the `make_seq_for_tok()` function to create a `seq_for_tok.txt` file with the following format:

```
CASSLLPGQGDGYTF
CASSLSGTHTGQETQYF
CASSNGLAGRVEQFF
CASSPDGGAYEQYF
...
KKAPAGPSL
YLFPGPVYV
FVYYPPGFRQILNY
MFLARAIVF
PYWVETITTTNNA
TRRVLLIVNGRVVR
RQLARLGMC
```

Train the tokenizer using the `train_bpe_tokenizer(vocab_size, seq_for_tok, outdir)` function from `tokenizer.py`. You can experiment with different vocabulary sizes based on your needs. Both `seq_for_tok.txt` and the trained tokenizer are available at https://zenodo.org/records/14286754.

## Training a GPT2 Model
With your pseudo-labeled dataset and tokenizer ready, you can now train a GPT-2 model to generate epitope sequences based on TCR sequences (CDR3b). We've adapted `run_clm_no_trainer.py` from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py, with modifications to tokenization, preprocessing, and batching. Use `scripts/train_eval_infer_EpiGen.sh`:

```
### --------------------------------------------------------------
### Train GPT-2-small architecture
### You need a trained tokenizer and config under regaler/EpiGen
### --------------------------------------------------------------
accelerate launch epigen/run_clm_no_trainer.py \
    --model_name_or_path gpt2-small \
    --train_file data/train.csv \
    --validation_file data/val.csv \
    --per_device_train_batch_size 512 \
    --per_device_eval_batch_size 512 \
    --num_train_epochs 100 \
    --tokenizer_name regaler/EpiGen \
    --checkpointing_steps epoch \
    --with_tracking \
    --report_to wandb \
    --gradient_accumulation_steps 1 \
    --gpt2_small \
    --output_dir 241205_example_run
```

This will generate model checkpoints in `241205_example_run/epoch_{i}/`. Training times vary:
- GPT2-small model with pseudo-labeled data: <4 hours using 4 NVIDIA L40S GPUs
- GPT2-medium model with intermediate dataset: >8 days
- Adding MHC information increases training time further

## Evaluating Binding Affinity
Generated epitopes should demonstrate high binding affinity with query TCRs. Use the Robust Affinity Predictor to measure binding affinity and compare it against 100 randomly sampled epitopes and TCRs. For each TCR, different epitopes are sampled from a large pool of unpaired epitopes. Calculate the percentile rank of binding affinity for each TCR-epitope pair using the `AffinityEvaluator` class from `epigen/eval/evaluate.py`.

Prediction files should contain these columns: tcr, epitope, pred_0, pred_1, ... where pred_{i} represents generated epitopes. Note that the `epitope` column is not used in this evaluation:

```
for dataset in ['VDJdb']:
    evaluator = AffinityEvaluator(
        pred_csvs=[
            f"predictions/240625_random/{dataset}/random_pred.csv",
            f"predictions/240625_knn/{dataset}/knn_pred.csv",
            f"predictions/241107_EpiGen/{dataset}/processed_{dataset}_test.csv"
        ],
        pmhc_data="data/240606_unique_peptides/peptides_for_neg.csv",
        outdir=f"figures/{dataset}",
        topk_values=[1, 5, 10, 20],
        pmhc_weight="tabr_bert_fork/output/pmhc_models_peptide_only/pmhc_model_e100.pt",
        tcr_weight="tabr_bert_fork/model/tcr_model.pt",
        model_weights=[
            "tabr_bert_fork/output/240612_rand_tcr_pmhc_1/tcr_pep_e58.pt",
            "tabr_bert_fork/output/240612_rand_tcr_pmhc_2/tcr_pep_e72.pt",
            "tabr_bert_fork/output/240612_rand_tcr_pmhc_3/tcr_pep_e60.pt",
            "tabr_bert_fork/output/240612_rand_tcr_pmhc_4/tcr_pep_e50.pt",
            "tabr_bert_fork/output/240612_rand_tcr_pmhc_5/tcr_pep_e50.pt",
        ]
    )
    evaluator.eval()
```

## Evaluation using InterfaceAnalyzer
To evaluate structural properties:

1. Use `convert_pred_to_tcrmodel2_format()` to convert prediction files for **TCRmodel2** (https://github.com/piercelab/tcrmodel2?tab=readme-ov-file) structure prediction.
   - Install **TCRmodel2** first (requires >700GB disk space due to **AlphaFold2** database)
   - This will generate **pdb** files

2. Apply **Rosetta Relax** operation on the pdb files:
   - Install **rosetta** from https://docs.rosettacommons.org/demos/latest/tutorials/install_build/install_build (license required)
   - Use `{{ROSETTA}}/source/bin/relax.static.linuxgccrelease` for relaxation

3. Use **InterfaceAnalyzer** (`rosetta.binary.linux.release-371/main/source/bin/InterfaceAnalyzer.static.linuxgccrelease`) to measure structural properties
   - Refer to `scripts/analyze_rosetta.py` for implementation details
