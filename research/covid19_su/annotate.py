# Standard library imports
import os
from itertools import product, combinations
from multiprocessing import Pool, cpu_count
from pathlib import Path

# Third-party imports
import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import gaussian_kde
from umap import UMAP

# Local imports
from research.covid19_su.utils import *


def annotation_wrapper(data_dir, pred_csv, epi_db_path, obs_cache=None, outdir=None, annotate_corona=False, match_method='levenshtein'):
    # A wrapper function to run series of annotations
    assert match_method in ['levenshtein', 'substring']
    adata = read_all_data(data_dir, gex_cache="covid19_su/gex_cache/cd8_gex.h5ad", obs_cache=obs_cache, use_multiprocessing=False, filtering=False)

    adata = annotate_cdr3(adata, data_dir, outdir=outdir, save_cdr3=False)
    adata = annotate_cdr3(adata, data_dir, outdir=outdir, save_cdr3=True)
    return

    adata = annotate_wos(adata, demographics_csv="covid19_su/data/demographics.csv", outdir=outdir)
    adata = annotate_cell_type(adata, outdir=outdir)
    return

    adata = insert_epitope_info(adata, pred_csv, outdir=outdir)
    adata = annotate_covid19_associated_epitopes(
        adata, epi_db_path, method=match_method, threshold=1, top_k=32, annotate_corona=annotate_corona, outdir=f"{outdir}/{match_method}")

    return adata


def annotate_cdr3(adata, data_dir, outdir=None, save_cdr3=False):
    print("Read CD8 TCR datasets from the txt files..")
    tcr_files = []
    for file in os.listdir(data_dir):
        if 'cd8' in file:
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path, delimiter='\t')
            tcr_files.append(df)

    # Concatenate all DataFrames and reset the index
    df_tcr = pd.concat(tcr_files, ignore_index=True)
    df_tcr = df_tcr.rename(columns={'Unnamed: 0': 'cell_barcode'})

    if save_cdr3 and outdir is not None:
        # Filter for single pair and select TRB_1_cdr3
        cdr3_series = df_tcr[df_tcr['chain_pairing'] == 'Single pair']['TRB_1_cdr3']

        # Save the original CDR3 series
        cdr3_series.to_csv(f"{outdir}/cdr3.csv", index=False, header=True)
        print(f"{outdir}/cdr3.csv was saved.")

        # Convert the series to a DataFrame and add the label column
        cdr3_df = cdr3_series.to_frame(name='text')
        cdr3_df['label'] = 'AAAAA'

        # Save the formatted DataFrame
        cdr3_df.to_csv(f"{outdir}/cdr3_formatted.csv", index=False)
        print(f"{outdir}/cdr3_formatted.csv was saved.")
        exit(0)

    print("Annotate the TCR information to adata..")
    # Merge the CDR3b sequence to GEX obs by cell_id
    obs_df = adata.obs
    merged_df = obs_df.merge(df_tcr, how='left', left_index=True, right_on='cell_barcode')
    # Set the index of the merged DataFrame to the original index (cell_id)
    merged_df.set_index('cell_barcode', inplace=True)
    # Assign the merged DataFrame back to mdata['gex'].obs
    adata.obs = merged_df
    if outdir is None:
        outdir = "gex_obs"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    adata.obs.to_csv(f"{outdir}/cdr3_added.csv", index=False)
    print(f"{outdir}/cdr3_added.csv was saved. ")
    return adata


def annotate_wos(adata, demographics_csv, outdir=None):
    def parse_filename(subject_id, prefix=None):
        pre = subject_id.split("-")[0]
        number = int(pre.split("INCOV")[1])
        order = int(subject_id.split("-")[1])  # 1 or i2
        filename = f"heathlab_dc_9_17_pbmc_gex_library_{number}_{order}.txt"
        if prefix:
            filename = prefix + filename
        return filename

    df = pd.read_csv(demographics_csv)
    df['filename'] = df['Sample ID'].apply(lambda x: parse_filename(x, prefix="data/covid19/"))
    obs_df = adata.obs

    merged_df = obs_df.merge(df, how='left', left_on='source_file', right_on='filename', indicator=True)
    # Drop the 'tcr' and '_merge' columns
    merged_df = merged_df.drop(columns=['cell_barcode_y', '_merge', 'filename'])
    merged_df = merged_df.rename(columns={'cell_barcode_x': 'cell_barcode'})
    # Stringify
    merged_df['Who Ordinal Scale'] = merged_df['Who Ordinal Scale'].apply(lambda x: str(x))
    if outdir is None:
        outdir = "gex_obs"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    # Save merged_df
    merged_df.to_csv(f"{outdir}/wos_added.csv", index=False)
    print(f"{outdir}/wos_added.csv was saved. ")
    merged_df.set_index('cell_barcode', inplace=True)
    adata.obs = merged_df
    return adata


def annotate_cell_type(adata, outdir=None):
    all_signature_genes = list(set([gene for genes in SIGNATURE_GENES.values() for gene in genes]))

    # Function to add signature genes to highly variable genes
    def add_signature_genes_to_hvg(adata, sig_genes, n_top_genes):
        # Get the names of the current highly variable genes
        current_hvg = adata.var.index[adata.var.highly_variable]

        # Add signature genes to the highly variable genes
        additional_hvg = [gene for gene in sig_genes if gene in adata.var_names and gene not in current_hvg]
        adata.var.loc[additional_hvg, 'highly_variable'] = True
        print(f"Added {len(additional_hvg)} signature genes to the {n_top_genes} highly variable genes.")

    def save_fig(fig, path):
        plt.savefig(path, format='pdf')
        print(f"{path} was saved. ")
        plt.close()

    os.makedirs(outdir, exist_ok=True)

    # Preprocessing
    sc.pp.normalize_total(adata, target_sum=1e6)  # CPM normalization
    sc.pp.log1p(adata)  # log1p transformation
    adata.layers['log1p'] = adata.X.copy()

    # Hyperparameters to sweep
    n_top_genes_list = [1000, 2000]
    n_pcs_list = [30, 50]
    n_neighbors_list = [10, 15]
    resolution_list = [1.0]

    # Iterate through hyperparameter combinations
    for n_top_genes, n_pcs, n_neighbors, resolution in product(n_top_genes_list, n_pcs_list, n_neighbors_list, resolution_list):
        print(f"Processing: n_top_genes={n_top_genes}, n_pcs={n_pcs}, n_neighbors={n_neighbors}, resolution={resolution}")
        outdir2 = f"{outdir}/hp_ntopgenes_{n_top_genes}_npcs_{n_pcs}_nn_{n_neighbors}"
        Path(outdir2).mkdir(parents=True, exist_ok=True)

        # Create a copy of the AnnData object
        adata_copy = adata.copy()

        # Highly variable genes
        sc.pp.highly_variable_genes(adata_copy, n_top_genes=n_top_genes)
        add_signature_genes_to_hvg(adata_copy, all_signature_genes, n_top_genes)

        # PCA
        sc.tl.pca(adata_copy, svd_solver='arpack', n_comps=n_pcs)
        fig = sc.pl.pca(adata_copy, return_fig=True)
        save_fig(fig, f"{outdir2}/pca_plot_ntopgenes_{n_top_genes}_npcs_{n_pcs}.pdf")
        fig = sc.pl.pca_variance_ratio(adata_copy, n_pcs=n_pcs, log=True)
        save_fig(fig, f"{outdir2}/pca_var_ratio_ntopgenes_{n_top_genes}_npcs_{n_pcs}.pdf")

        # Neighborhood graph and UMAP
        sc.pp.neighbors(adata_copy, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.umap(adata_copy)
        # Save UMAP coordinates to obs dataframe
        adata_copy.obs['UMAP1'] = adata_copy.obsm['X_umap'][:, 0]
        adata_copy.obs['UMAP2'] = adata_copy.obsm['X_umap'][:, 1]
        fig = sc.pl.umap(adata_copy, size=2, return_fig=True)
        save_fig(fig, f"{outdir2}/umap_uncolored_ntopgenes_{n_top_genes}_npcs_{n_pcs}_nneighbors_{n_neighbors}.pdf")
        sc.tl.leiden(adata_copy, resolution=resolution)
        fig = sc.pl.umap(adata_copy, color=['leiden'], return_fig=True)
        save_fig(fig, f'{outdir2}/umap_leiden_ntopgenes_{n_top_genes}_npcs_{n_pcs}_nneighbors_{n_neighbors}_resolution_{resolution}.pdf')

        # Calculate signature scores
        for sig_name, sig_genes in SIGNATURE_GENES.items():
            print(f"Processing sig_name: {sig_name}, sig_genes: {sig_genes}")
            gene_indices = [adata_copy.var_names.get_loc(gene) for gene in sig_genes if gene in adata_copy.var_names]
            if gene_indices:
                sig_scores = np.sum(adata_copy.layers['log1p'][:, gene_indices], axis=1)
                adata_copy.obs[f'{sig_name}_score'] = (sig_scores - np.min(sig_scores)) / (np.max(sig_scores) - np.min(sig_scores))
        print("scRNA CD8+ T cell signature scores calculations were done!")

        # Calculate mean expression of signature genes in each cluster
        cluster_means = {}
        for sig_name, sig_genes in SIGNATURE_GENES.items():
            cluster_means[sig_name] = adata_copy[:, sig_genes].to_df().groupby(adata_copy.obs['leiden']).mean()

        # Visualize mean expression of signature genes in each cluster
        plt.figure(figsize=(20, 4 * len(SIGNATURE_GENES)))
        for i, (sig_name, means) in enumerate(cluster_means.items()):
            plt.subplot(len(SIGNATURE_GENES), 1, i+1)
            sns.heatmap(means, cmap='YlOrRd', annot=True, fmt='.2f', cbar_kws={'label': 'Mean Expression'})
            plt.title(f'Mean Expression of {sig_name.capitalize()} Signature Genes by Cluster')
            plt.ylabel('Cluster (leiden)')
        plt.tight_layout()
        plt.savefig(f'{outdir2}/signature_genes_heatmap.pdf', format='pdf')
        plt.close()

        print("\nAnalysis complete. Visualizations saved as PNG files.")

        # Calculate mean expression and percentage of cells expressing each gene per cluster
        def calculate_dot_plot_stats(adata, genes, groupby='leiden'):
            """Calculate statistics for dot plot visualization."""
            # Create empty matrices for means and percentages
            n_groups = len(adata.obs[groupby].unique())
            n_genes = len(genes)
            means = np.zeros((n_groups, n_genes))
            pcts = np.zeros((n_groups, n_genes))

            for ig, group in enumerate(adata.obs[groupby].unique()):
                for ig2, gene in enumerate(genes):
                    if gene in adata.var_names:
                        _adata = adata[adata.obs[groupby] == group, gene]
                        means[ig, ig2] = np.mean(_adata.X)
                        pcts[ig, ig2] = np.sum(_adata.X > 0) / len(_adata.X)

            return means, pcts

        # Collect all signature genes in the desired order
        desired_order = ['naive', 'memory', 'cytotoxic', 'exhaustion', 'proliferation']
        all_sig_genes = []
        var_group_labels = []
        var_group_positions = []
        position = 0

        for sig_name in desired_order:
            sig_genes = SIGNATURE_GENES[sig_name]
            all_sig_genes.extend(sig_genes)  # Add genes in order
            var_group_labels.append(sig_name)
            var_group_positions.append(position)
            position += len(sig_genes)

        # Calculate mean expression and percentage of cells expressing each gene
        means, pcts = calculate_dot_plot_stats(adata_copy, all_sig_genes)

        # Create var_names categories for grouping genes by signature
        var_group_labels = []
        var_group_positions = []
        position = 0
        for sig_name, sig_genes in SIGNATURE_GENES.items():
            var_group_labels.append(sig_name)
            var_group_positions.append(position)
            position += len(sig_genes)

        # Create a new AnnData object for dot plot
        adata_dot = sc.AnnData(X=means.T,
                              var=pd.DataFrame(index=adata_copy.obs['leiden'].unique()),
                              obs=pd.DataFrame(index=all_sig_genes))

        adata_dot = adata_dot.transpose()
        adata_dot.obs['leiden'] = adata_dot.obs.index

        # Plot using scanpy's dot plot function
        plt.figure(figsize=(15, 10))
        sc.pl.dotplot(adata_dot,
                      var_names=all_sig_genes,
                      groupby='leiden',
                      standard_scale='var',
                      title='Gene Signatures across Clusters',
                      show=False)  # Use show=False instead of return_fig=True
        plt.savefig(f'{outdir2}/signature_genes_dotplot.pdf', bbox_inches='tight', format='pdf')
        plt.close()

        # Visualize the clusters by WOS
        print("Visualize the clusters by WOS")
        print("Stringify Who Ordinal Scale..")
        adata_copy.obs['Who Ordinal Scale'] = adata_copy.obs['Who Ordinal Scale'].apply(lambda x: str(x))

        def plot_umap_wos_with_density(adata, wos_patterns, outdir):
            # Set up the figure
            fig, axs = plt.subplots(2, 2, figsize=(20, 20))
            axs = axs.flatten()

            # Get UMAP coordinates
            umap_coords = pd.DataFrame(adata.obsm['X_umap'], columns=['UMAP1', 'UMAP2'])
            umap_coords['Who Ordinal Scale'] = adata.obs['Who Ordinal Scale'].values

            # Create a meshgrid for density estimation
            xx, yy = np.mgrid[umap_coords.UMAP1.min():umap_coords.UMAP1.max():100j,
                              umap_coords.UMAP2.min():umap_coords.UMAP2.max():100j]

            for i, wos_pattern in enumerate(wos_patterns):
                ax = axs[i]

                # Plot light gray background points
                sns.scatterplot(data=umap_coords, x='UMAP1', y='UMAP2', color='lightgray',
                                alpha=0.1, s=1, ax=ax)

                # Filter for the current wos_pattern
                mask = umap_coords['Who Ordinal Scale'].isin(wos_pattern)
                subset = umap_coords[mask]

                if not subset.empty:
                    # Perform kernel density estimation
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    kernel = gaussian_kde(subset[['UMAP1', 'UMAP2']].values.T)
                    f = np.reshape(kernel(positions).T, xx.shape)

                    # Plot the density heatmap
                    im = ax.imshow(np.rot90(f), cmap='viridis', extent=[umap_coords.UMAP1.min(), umap_coords.UMAP1.max(),
                                                                        umap_coords.UMAP2.min(), umap_coords.UMAP2.max()],
                                   aspect='auto', alpha=0.8)

                    # Add colorbar
                    plt.colorbar(im, ax=ax)

            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(f'{outdir}/umap_wos_ntopgenes_{n_top_genes}_npcs_{n_pcs}_nn_{n_neighbors}.png', dpi=300, bbox_inches='tight')
            plt.close()

        plot_umap_wos_with_density(adata_copy, WOS_PATTERNS, outdir2)
        # Save results
        adata_copy.obs.to_csv(f'{outdir2}/cell_metadata.csv', index=False)
        adata_copy.var.to_csv(f'{outdir2}/gene_metadata.csv', index=False)
        print(f'{outdir2}/cell_metadata.csv')
        print(f'{outdir2}/gene_metadata.csv')

    return adata


def insert_epitope_info(adata, pred_csv, outdir=None):
    # Insert the EpiGen prediction to adata
    df = pd.read_csv(pred_csv)
    obs_df = adata.obs.copy()

    # Ensure the cdr3 column in adata.obs and tcr column in df are strings
    obs_df['TRB_1_cdr3'] = obs_df['TRB_1_cdr3'].astype(str)
    df['tcr'] = df['tcr'].astype(str)

    # Make TCR entries unique in the prediction dataframe
    df = df.drop_duplicates(subset='tcr', keep='first')

    # The Index `cell_id` becomes a column
    obs_df = obs_df.reset_index()

    # Perform the merge
    merged_df = obs_df.merge(df, how='left', left_on='TRB_1_cdr3', right_on='tcr', indicator=True)
    # Handle unmatched entries
    unmatched_columns = df.columns.drop('tcr')
    for col in unmatched_columns:
        if col not in merged_df.columns:
            merged_df[col] = np.nan

    # Drop the 'tcr' and '_merge' columns
    merged_df = merged_df.drop(columns=['tcr', '_merge'])

    # Set the 'cell_id' column back as the index
    merged_df.set_index('cell_barcode', inplace=True)
    # Assign the merged DataFrame back to adata.obs
    adata.obs = merged_df

    # Save the updated obs DataFrame
    if outdir is None:
        outdir = "gex_obs"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    adata.obs.to_csv(f"{outdir}/epitopes_added.csv", index=True)
    print(f"{outdir}/epitopes_added.csv was saved.")

    return adata


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


def annotate_covid19_associated_epitopes(adata, epi_db_path, method='levenshtein', threshold=1, top_k=1, annotate_corona=False, outdir=None):
    """
    Annotate cells in adata for covid19 association by querying epitope databases.
    Parameters:
    - adata: The AnnData object containing the cell data.
    - epi_db_path: Path to the CSV file containing the reference epitopes.
    - method: The matching method to use, either 'levenshtein' or 'substring'.
    - threshold: The maximum Levenshtein distance for a match (only used for 'levenshtein' method).
    """
    # Uniformize sequence lengths in pred_0 to pred_9 to a maximum of 9 characters
    for i in range(top_k):  # Adjust the range as needed
        pred_column_name = f'pred_{i}'
        adata.obs[pred_column_name] = adata.obs[pred_column_name].apply(lambda x: x[:9] if isinstance(x, str) else x)

    # Load the epitope database
    epi_db = pd.read_csv(epi_db_path)

    for i in range(top_k):  # Adjust the range as needed
        print(f"Start querying pred_{i}")
        pred_column_name = f'pred_{i}'
        if not annotate_corona:  # normally, just COVID-19
            match_column_name = f'match_{i}'
            ref_epitope_column_name = f'ref_epitope_{i}'
            ref_protein_column_name = f'ref_protein_{i}'
        else:
            match_column_name = f'corona_{i}'
            ref_epitope_column_name = f'corona_epitope_{i}'
            ref_protein_column_name = f'corona_protein_{i}'

        # Run the matching process
        results = process_predictions(adata.obs[pred_column_name], epi_db, threshold, method)

        # Unpack the results into separate columns
        adata.obs[match_column_name], adata.obs[ref_epitope_column_name], adata.obs[ref_protein_column_name] = zip(*results)

    if outdir is None:
        outdir = "gex_obs"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    if not annotate_corona:  # normal, just COVID-19
        adata.obs.to_csv(f"{outdir}/gex_obs_EpiGen_annotated.csv", index=False)
        print(f"{outdir}/gex_obs_EpiGen_annotated.csv was saved. ")
    else:
        adata.obs.to_csv(f"{outdir}/coronavirus_annotated.csv", index=False)
        print(f"{outdir}/coronavirus_annotated.csv was saved. ")
    return adata


def visualize_signature_dotplot(adata, cell_metadata_path, gene_metadata_path, signature_genes_dict, outdir, groupby='leiden'):
    """
    Create a dot plot visualization from saved metadata files.
    Use saved files after running annotate_cell_type() function

    Parameters:
    -----------
    cell_metadata_path : str
        Path to the saved cell metadata CSV
    gene_metadata_path : str
        Path to the saved gene metadata CSV
    signature_genes_dict : dict
        Dictionary of signature names and their associated genes
    outdir : str
        Path to save the output dot plot
    groupby : str, optional
        Column name in cell_metadata to group cells by (default: 'leiden')
    """
    # Load the metadata
    cell_metadata = pd.read_csv(cell_metadata_path)
    gene_metadata = pd.read_csv(gene_metadata_path)

    # Create a basic AnnData object
    # adata = sc.AnnData(X=np.zeros((len(cell_metadata), len(gene_metadata))))
    adata.obs = cell_metadata
    # adata.var = gene_metadata

    # Calculate mean expression and percentage of cells expressing each gene per cluster
    def calculate_dot_plot_stats(adata, genes, groupby='leiden'):
        """Calculate statistics for dot plot visualization."""
        # Create empty matrices for means and percentages
        n_groups = len(adata.obs[groupby].unique())
        n_genes = len(genes)
        means = np.zeros((n_groups, n_genes))
        pcts = np.zeros((n_groups, n_genes))

        for ig, group in enumerate(adata.obs[groupby].unique()):
            for ig2, gene in enumerate(genes):
                if gene in adata.var_names:
                    _adata = adata[adata.obs[groupby] == group, gene]
                    means[ig, ig2] = np.mean(_adata.X)
                    pcts[ig, ig2] = np.sum(_adata.X > 0) / len(_adata.X)

        return means, pcts

    # Preprocessing
    sc.pp.normalize_total(adata, target_sum=1e6)  # CPM normalization
    sc.pp.log1p(adata)  # log1p transformation
    adata.layers['log1p'] = adata.X.copy()

    # Collect all signature genes in the desired order
    desired_order = ['naive', 'memory', 'cytotoxic', 'exhaustion', 'proliferation']
    all_sig_genes = []
    var_group_labels = []
    var_group_positions = []
    position = 0

    for sig_name in desired_order:
        sig_genes = SIGNATURE_GENES[sig_name]
        all_sig_genes.extend(sig_genes)  # Add genes in order
        var_group_labels.append(sig_name)
        var_group_positions.append(position)
        position += len(sig_genes)

    # Calculate mean expression and percentage of cells expressing each gene
    means, pcts = calculate_dot_plot_stats(adata, all_sig_genes)

    # Create var_names categories for grouping genes by signature
    var_group_labels = []
    var_group_positions = []
    position = 0
    for sig_name, sig_genes in SIGNATURE_GENES.items():
        var_group_labels.append(sig_name)
        var_group_positions.append(position)
        position += len(sig_genes)

    # Create a new AnnData object for dot plot
    adata_dot = sc.AnnData(X=means.T,
                            var=pd.DataFrame(index=adata.obs['leiden'].unique()),
                            obs=pd.DataFrame(index=all_sig_genes))

    adata_dot = adata_dot.transpose()
    adata_dot.obs['leiden'] = adata_dot.obs.index

    # Plot using scanpy's dot plot function
    plt.figure(figsize=(15, 10))
    sc.pl.dotplot(adata_dot,
                    var_names=all_sig_genes,
                    groupby='leiden',
                  categories_order=['1', '5', '3', '4', '0', '6', '2'],
                    standard_scale='var',
                    title='Gene Signatures across Clusters',
                    show=False)  # Use show=False instead of return_fig=True
    plt.savefig(f'{outdir}/signature_genes_dotplot.pdf', bbox_inches='tight', format='pdf')
    plt.close()
