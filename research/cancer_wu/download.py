# Standard library imports
import gzip
import os
import shutil
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

# Third-party imports
import anndata
import pandas as pd
import rpy2.robjects as ro
import scanpy as sc
import scirpy as ir
from rpy2.robjects import pandas2ri
from tqdm import tqdm


# Activate the pandas conversion
pandas2ri.activate()

def download_file(url, local_filename):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True,
                           desc=local_filename) as t:
        urlretrieve(url, local_filename, reporthook=t.update_to)
    print(f"\nDownloaded {local_filename}")

def extract_tar_file(tar_filename, extract_path="."):
    with tarfile.open(tar_filename, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print(f"Extracted {tar_filename}")

def extract_gz_file(gz_filename, output_filename=None):
    """
    Extract a gzipped file (.gz), supporting both .rds.gz and .txt.gz files.

    Args:
        gz_filename (str): Path to the gzipped file
        output_filename (str, optional): Custom output filename. If None, automatically determined.

    Returns:
        bool: True if extraction successful, False otherwise
    """
    try:
        # Validate input file
        if not os.path.exists(gz_filename):
            raise FileNotFoundError(f"Input file not found: {gz_filename}")

        # Handle output filename
        if output_filename is None:
            # Use pathlib for robust path handling
            path = Path(gz_filename)
            if not path.name.endswith('.gz'):
                raise ValueError(f"Input file does not have .gz extension: {gz_filename}")

            # Remove .gz and keep original extension (.rds or .txt)
            output_filename = str(path.with_suffix('').absolute())

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)

        # Extract the gzipped file
        with gzip.open(gz_filename, 'rb') as f_in:
            with open(output_filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out, length=1024*1024)  # 1MB chunks for efficiency

        print(f"Successfully extracted {gz_filename} to {output_filename}")
        return True

    except gzip.BadGzipFile:
        print(f"Error: {gz_filename} is not a valid gzip file")
        return False
    except FileNotFoundError as e:
        print(str(e))
        return False
    except ValueError as e:
        print(str(e))
        return False
    except Exception as e:
        print(f"Error extracting {gz_filename}: {str(e)}")
        return False

def convert_rds_to_anndata(rds_filename):
    try:
        # Check if file exists
        if not os.path.exists(rds_filename):
            raise FileNotFoundError(f"File not found: {rds_filename}")

        # Initialize R converter
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()

        # Read RDS file
        print(f"Reading {rds_filename}...")
        ro.r(f'data <- readRDS("{rds_filename}")')

        # Check the class of R object
        r_class = ro.r('class(data)')
        print(f"R object class: {r_class[0]}")

        # Convert to pandas DataFrame
        print("Converting to pandas DataFrame...")
        if r_class[0] in ['matrix', 'data.frame']:
            # Transpose the data in R before converting to DataFrame
            ro.r('data <- t(data)')  # Transpose in R
            r_df = ro.r('as.data.frame(data)')
            df = pandas2ri.rpy2py(r_df)
        else:
            raise ValueError(f"Unsupported R object class: {r_class[0]}")

        # Convert to AnnData
        print("Creating AnnData object...")
        adata = sc.AnnData(df)

        # Add some basic metadata
        adata.uns['original_file'] = rds_filename
        adata.uns['conversion_source'] = 'rds'

        print(f"Successfully converted {rds_filename} to AnnData:")
        print(f"Shape: {adata.shape}")
        return adata

    except Exception as e:
        print(f"Error converting {rds_filename}: {str(e)}")
        raise
    finally:
        # Clean up R objects
        ro.r('rm(data)')
        ro.r('gc()')

def load_metadata(metadata_filename):
    """
    Load metadata from a tab-delimited text file into an AnnData object.

    Args:
        metadata_filename (str): Path to the metadata file

    Returns:
        anndata.AnnData: Metadata as an AnnData object
    """
    try:
        # First read as pandas DataFrame to handle mixed data types
        df = pd.read_csv(metadata_filename,
                        sep='\t',
                        index_col=0)  # First column as index
    except Exception as e:
        print(f"Error loading metadata from {metadata_filename}: {str(e)}")
        raise
    return df


def download_and_preprocess(outdir="cancer_wu/data", input_file="cancer_wu/data_links.txt"):
    samples = ['CN1', 'CT2', 'EN3', 'ET3', 'LB6', 'LN3', 'LN6', 'LT3', 'LT6', 'RB2', 'RN2', 'RT2',
               'CN2', 'EN1', 'ET1', 'LN1', 'LN4', 'LT1', 'LT4', 'RB3', 'RN3', 'RT3',
               'CT1', 'EN2', 'ET2', 'LN2', 'LN5', 'LT2', 'LT5', 'RB1', 'RN1', 'RT1']
    # Get the files by referring to https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139555
    with open(input_file, "r") as f:
        remote_filenames = f.readlines()
    remote_filenames = [x.strip() for x in remote_filenames]
    local_filenames = [f"{outdir}/{os.path.basename(filename)}" for filename in remote_filenames]
    Path(outdir).mkdir(parents=True, exist_ok=True)
    Path(f"{outdir}/raw").mkdir(parents=True, exist_ok=True)

    for sample in samples:
        if os.path.isdir(f"{outdir}/{sample}"):
            data_exists = True
        else:
            data_exists = False

    if data_exists:
        print(f"Cancer wu data already exists in {outdir}")
        return

    # Download the necessary data
    for filename, local_filename in zip(remote_filenames, local_filenames):
        download_file(filename, local_filename)

    # Extract the files
    for gz_file in os.listdir(outdir):
        if gz_file.endswith(".gz"):
            extract_gz_file(gz_filename=f"{outdir}/{gz_file}")
            os.remove(f"{outdir}/{gz_file}")

    # Convert RDS to AnnData
    rds_file = [f"{outdir}/{x}" for x in os.listdir(outdir) if x.endswith(".rds")][0]
    try:
        adata = convert_rds_to_anndata(rds_file)
        print(adata)
    except Exception as e:
        print(f"Conversion failed: {str(e)}")

    # Load metadata
    meta_filename = [f"{outdir}/{x}" for x in os.listdir(outdir) if x.endswith("metadata.txt")][0]
    metadata = load_metadata(meta_filename)
    adata.obs = metadata

    # Save the processed / converted data
    adata.write_h5ad(f"{outdir}/GSE139555_tcell_integrated.h5ad")
    metadata.to_csv(f"{outdir}/GSE139555%5Ftcell%5Fmetadata.txt")
    print(f"{outdir}/GSE139555_tcell_integrated.h5ad")
    print(f"{outdir}/GSE139555%5Ftcell%5Fmetadata.txt")

    # Structure the data by sample
    for sample in samples:
        Path(f"{outdir}/{sample}").mkdir(parents=True, exist_ok=True)

    def _parse(x):
        return x.split(".")[0].split("2D")[1].upper()

    files = [x for x in os.listdir(outdir) if x.startswith("GSM")]
    for file in files:
        sample = _parse(f"{outdir}/{file}")
        os.rename(f"{outdir}/{file}", f"{outdir}/{sample}/{file}")
