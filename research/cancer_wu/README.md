## Cancer Dataset Analysis Pipeline

### Dataset Overview
The Wu et al. dataset ([Nature publication](https://www.nature.com/articles/s41586-020-2056-8)) comprises paired single-cell RNA and TCR sequencing data from 14 cancer patients:
- 10 patients: Samples from tumor tissue and normal adjacent tissue
- 4 patients: Samples from tumor tissue, normal adjacent tissue, and peripheral blood

### Data Preparation

1. Download and preprocess the sequencing data:
```python
# In EpiGen/
python scripts/run_cancer_wu.py --download
```
This command executes `download_and_preprocess()` from `cancer_wu/download.py` and saves datasets in `cancer_wu/data`.

2. Download reference data:
- Obtain the curated tumor-associated epitopes database from [Zenodo](https://zenodo.org/records/14344830)
- The `tumor_associated_epitopes.csv` file serves as the reference database for epitope queries

### EpiGen Analysis
EpiGen is a generative model that predicts cognate epitope sequences from TCR sequences. Its primary function in this pipeline is to classify TCRs into:
- Tumor-associated (PA) TCRs
- Background (NA) TCRs

This classification enables comprehensive comparative analyses between the two groups.

### T Cell Annotation Process

1. Generate epitope sequences:
   - Format CDR3b sequences using `save_cdr3_formatted()` from `cancer_wu/utils.py`:
   ```python
   python scripts/run_cancer_wu.py --utils
   ```
   - Run inference using `scripts/train_eval_infer_EpiGen.sh` (adjust file paths and parameters as needed)
   - Store the generated epitopes CSV file in an appropriate directory (e.g., `cancer_wu/predictions/pred.csv`)

2. Annotate the cancer dataset (h5ad file):
   - Map CDR3b sequences to T cells using `annotate_cdr3_scirpy()`
   - Add epitope information with `insert_epitope_info()`
   - Label tumor-associated epitopes using `annotate_tumor_associated_epitopes()`
   - Add site pattern annotations using `annotate_sites()`

3. Execute the annotation pipeline:
```python
python scripts/run_cancer_wu.py --annotate
```

### Analysis Components

The pipeline includes four main analytical approaches comparing PA and NA T cells:

1. **PA Ratio Analysis**
   - Quantifies Phenotype-Associated T cells within specific repertoire subgroups
   - Enables comparison of tumor-associated T cell proportions across site patterns and cell types

2. **CE (Clonal Expansion) Ratio Analysis**
   - Evaluates clone sizes of PA T cells
   - Provides comparative analysis against NA T cell expansion

3. **Gene Expression Analysis**
   - Performs differential gene expression analysis between PA and NA T cells
   - Identifies distinctive gene expression patterns in PA T cells
   - Requires per-patient raw data (accessed via `read_all_raw_data()`)

4. **Antigen Analysis**
   - Examines antigen sources of generated epitopes
   - Calculates tumor-associated ratios

Execute all analyses using:
```python
python scripts/run_cancer_wu.py --analyze
```

Note: Most analyses utilize `read_all_data()`, except for gene expression analysis which requires `read_all_raw_data()`. For detailed conceptual understanding, refer to the manuscript.
