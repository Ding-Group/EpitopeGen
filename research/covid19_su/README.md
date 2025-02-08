## COVID-19 Single-Cell Analysis Pipeline

This pipeline processes and analyzes single-cell paired RNA and TCR sequencing data from 139 donors, as published by Su et al. ([Cell, 2020](https://www.cell.com/cell/fulltext/S0092-8674(20)31444-6?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0092867420314446%3Fshowall%3Dtrue)). The dataset categorizes patients into four severity groups: healthy, mild, moderate, and severe. The analysis requires COVID-19-associated epitopes, which can be obtained from the [curated list on Zenodo](https://zenodo.org/records/14344830) or constructed independently using IEDB.

## Data Annotation Pipeline

### 1. Initial Data Processing
- Use `read_all_data()` from `covid19_su/annotate.py` with `filtering=True` to extract CD8+ T cells data
- Output: `matched_data.h5ad`
- Note: For subsequent runs, use `filtering=False` and specify the generated file in `gex_cache` for faster processing

### 2. CDR3 Sequence Annotation
- Execute `annotate_cdr3()` to add CDR3b sequences to each cell
- Processes CD8+ T cell files from `data_dir`
- Merges CDR3b information with GEX observations using cell_id
- Output: `gex_obs/cdr3_added.csv`

### 3. WHO Ordinal Scale (WOS) Annotation
- Apply `annotate_wos()` using demographics data
- Output: `gex_obs/wos_added.csv`

### 4. EpiGen Integration
1. Generate EpiGen input:
   - Run `annotate_cdr3()` with `save_cdr3=True`
2. Execute EpiGen inference
3. Incorporate results using `insert_epitope_info()`

### 5. Cell Type Annotation
- Use `annotate_cell_type()` for standard scRNA-seq analysis:
  - Data normalization
  - Log transformation
  - Highly variable gene selection
  - Signature gene addition
  - PCA and UMAP dimensionality reduction
- Hyperparameter optimization available for:
  - Number of highly variable genes
  - Principal component count
  - Neighbor count
- After parameter selection, copy `cell_metadata.csv` to `gex_obs/`

### 6. Epitope Phenotype Association
- Execute `annotate_covid19_associated_epitopes()`
- Matching methods:
  - Substring-based
  - Levenshtein distance-based
- Output: `gex_obs/gex_obs_EpiGen_annotated.csv`

## Analysis Methods

The analysis comprises four main components, which can be executed using `analysis_wrapper()` from `covid19_su/analyze.py`:

1. Phenotype-Association (PA) ratio analysis
2. Clonal Expansion analysis
3. Gene expression analysis
4. Antigen proportion analysis

For detailed conceptual explanations of these analyses, please refer to our manuscript.
