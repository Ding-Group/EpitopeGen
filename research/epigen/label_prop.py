# Standard library imports
import gc
import multiprocessing as mp
import os
import pickle
import random
import re
from datetime import datetime as dt
from functools import wraps
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# External imports
from tabr_bert_fork.bert_pmhc import BERT as pmhc_net
from tabr_bert_fork.bert_tcr import BERT as tcr_net
from tabr_bert_fork.tcr_pmhc_model import *  # Consider explicit imports instead of *
from research.epigen.utils import adheres_to_vocab, index_to_aa


class TCRPepDataset(Dataset):
    def __init__(self, tcr_seqs, tcr_feats, pep_seqs, pep_feats, num_candidate=10000):
        self.tcr_seqs = tcr_seqs
        self.tcr_feats = torch.tensor(np.array(tcr_feats), dtype=torch.float32)  # Convert list to numpy array first
        self.pep_seqs = pep_seqs
        self.pep_feats = torch.tensor(np.array(pep_feats), dtype=torch.float32)  # Convert list to numpy array first
        self.keys_pool = torch.arange(len(pep_seqs))  # Create a tensor of indices
        self.num_candidate = num_candidate

    def __len__(self):
        return len(self.tcr_seqs)

    def __getitem__(self, idx):
        tcr_seq = self.tcr_seqs[idx]
        tcr_feat = self.tcr_feats[idx].unsqueeze(0).expand((self.num_candidate, -1, -1))

        # Sample `num_candidate` random peptides
        keys = self.keys_pool[torch.randint(len(self.keys_pool), (self.num_candidate,))]  # Efficient random sampling
        pep_feat = self.pep_feats[keys]

        x = torch.cat([tcr_feat, pep_feat], dim=1)
        return x, tcr_seq, keys


class TCRPepSampler:
    """
    tcr_feat_pkl: str
        1 TCR feature pkl file
    pep_feat_root: str
        dir where ALL peptide feature pkl files are
    outdir: str:
        output directory
    """
    def __init__(self, tcr_feat_pkl, pep_feat_root, model_paths, outdir, tcr_chunk=4096, batch_size=16, num_candidate=10000, topk=256):
        self.tcr_feat_pkl = tcr_feat_pkl
        self.pep_feat_root = pep_feat_root
        self.outdir = self._init_dir(outdir, tcr_feat_pkl)
        self.model_mode = 'softmax'
        self.device = 'cuda'
        self.pmhc_maxlen = 18
        self.num_epi_files = 6

        self.tcr_chunk = tcr_chunk
        self.batch_size = batch_size
        self.softmax = nn.Softmax(dim=2)
        self.num_candidate = num_candidate
        self.topk = topk
        self.flush_size = self.tcr_chunk * 1
        self.tcr_data_idx = int(os.path.basename(tcr_feat_pkl)[:-4].split("_")[-1])  # ex) 17 in tcr_features_17.pkl
        self.tcr_seqs, self.tcr_feats, self.latest_tcr_index = self._read_tcrs(tcr_feat_pkl)
        self.pep_seqs, self.pep_feats = self._read_peptides(pep_feat_root)
        self.models = self._init_model(model_paths)

        self.pep_reset_index = int(self.latest_tcr_index / 31000)
        self._print_summary()

    def _print_summary(self):
        summary_attrs = [
            ("Output Directory", self.outdir),
            ("Model Mode", self.model_mode),
            ("Device", self.device),
            ("PMHC Max Length", self.pmhc_maxlen),
            ("TCR Chunk Size", self.tcr_chunk),
            ("Batch Size", self.batch_size),
            ("Number of Candidates", self.num_candidate),
            ("Top K", self.topk),
            ("Flush Size", self.flush_size),
            ("TCR Data Index", self.tcr_data_idx),
            ("Number of TCR Sequences", len(self.tcr_seqs)),
            ("Number of Peptide Sequences", len(self.pep_seqs)),
            ("Number of Models", len(self.models)),
            ("Latest TCR Index", self.latest_tcr_index),
            ("Number of pmhc feature files", self.num_epi_files)
        ]
        print("TCRPepSampler Summary:")
        for name, value in summary_attrs:
            print(f"{name}: {value}")

    def _read_tcrs(self, tcr_feat_pkl):
        with open(tcr_feat_pkl, "rb") as f:
            tcrs = pickle.load(f)
        tcr_seqs = list(tcrs.keys())
        tcr_feats = list(tcrs.values())
        # Start from where we left off
        latest_tcr_index = self._fetch_latest_index(self.tcr_data_idx)
        return tcr_seqs, tcr_feats, latest_tcr_index

    def _read_peptides(self, pep_feat_root):
        def get_postfix(filename):
            # Use regular expression to find numbers in the filename
            match = re.search(r'\d+', filename)
            return int(match.group()) if match else 0

        epi_files = sorted([x for x in os.listdir(pep_feat_root) if x.startswith("epi_features") and x.endswith("pkl")], key=get_postfix)
        epi_files = random.sample(epi_files, k=self.num_epi_files)
        peps = {}
        print("Reading peptide features..")
        for epi_pkl in tqdm(epi_files):
            with open(f"{pep_feat_root}/{epi_pkl}", "rb") as f:
                epi = pickle.load(f)
            for k in epi.keys():
                peps[k] = epi[k]
        pep_seqs = list(peps.keys())
        pep_feats = list(peps.values())
        return pep_seqs, pep_feats

    def _init_dir(self, outdir, tcr_data):
        # Get the current timestamp
        timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
        # Extract the TCR data index from the file name
        tcr_data_idx = os.path.splitext(os.path.basename(tcr_data))[0].split('_')[-1]
        outdir = f"{outdir}_tcr_data_{tcr_data_idx}_{timestamp}"
        # Create the directory with all parent directories if they don't exist
        Path(outdir).mkdir(parents=True, exist_ok=True)
        print(f"{outdir} was created.")
        return outdir

    def _init_model(self, model_paths):
        # Use list comprehension for conciseness and filter paths with start condition
        return [self._load_and_prepare_model(path) for path in model_paths]

    def _load_and_prepare_model(self, model_path):
        # Initialize and load the model only once in a separate method for clarity
        model = nn.DataParallel(tcr_pmhc(mode=self.model_mode, pmhc_maxlen=self.pmhc_maxlen))
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        if self.device == 'cuda':  # GPU
            model.cuda()
        model.eval()
        return model

    def sample(self):
        print(f"Start from TCR index: {self.latest_tcr_index}..")
        self.chunk_tcrs, self.chunk_peps = [], []  # accumulate results
        for tcr_idx in tqdm(range(self.latest_tcr_index, len(self.tcr_seqs), self.tcr_chunk)):
            tcr_slice = slice(tcr_idx, tcr_idx + self.tcr_chunk)
            current_chunk = (
                self.tcr_seqs[tcr_slice],
                self.tcr_feats[tcr_slice],
                self.pep_seqs,
                self.pep_feats
            )
            dataset = TCRPepDataset(*current_chunk)
            dataloader = DataLoader(dataset, batch_size=self.batch_size)
            for x, tcr_seqs, keys in tqdm(dataloader):
                if self.device == 'cuda':
                    x, keys = x.cuda(), keys.cuda()

                with torch.no_grad():
                    batch_preds = torch.mean(torch.stack(
                        [self.softmax(model(x).view(self.batch_size, self.num_candidate, 2))[:, :, 1]
                        for model in self.models]), dim=0)

                    sorted_vals, sorted_indices = torch.sort(batch_preds, dim=1, descending=True)
                    top_indices = sorted_indices[:, :self.topk]
                    pep_indices = torch.gather(keys, 1, top_indices)
                    self.chunk_tcrs.extend(tcr_seqs)
                    self.chunk_peps.append(pep_indices)

            if len(self.chunk_tcrs) >= self.flush_size or tcr_idx == len(self.tcr_seqs) - self.tcr_chunk:
                self.flush_current_stack(tcr_end_idx=tcr_idx + self.tcr_chunk)

    def flush_current_stack(self, tcr_end_idx):
        # Write the info in self.chunk_tcrs and self.chunk_peps into file and refresh it
        num_of_data = len(self.chunk_tcrs)
        self.chunk_peps = torch.cat(self.chunk_peps)  # (num_of_data, self.num_candidate)
        # Convert chunk_peps to numpy for indexing
        chunk_peps_np = self.chunk_peps.cpu().numpy()
        # Efficiently retrieve the peptide sequences
        tcr2peps = {self.chunk_tcrs[i]: [self.pep_seqs[idx] for idx in chunk_peps_np[i]] for i in range(num_of_data)}

        # Save information of tcr -> peptide sequences (dict) as a pkl file
        outfile = f"{self.outdir}/sampled_data_{tcr_end_idx}.pkl"
        with open(outfile, "wb") as f:  # Use "wb" for writing in binary mode
            pickle.dump(tcr2peps, f)

        print(f"{outfile} was saved. Flushing self.chunk_tcrs, self.chunk_peps..")
        self.chunk_tcrs, self.chunk_peps = [], []
        self.latest_tcr_index = tcr_end_idx
        # Reset / re-read the peptide data periodically
        if int(tcr_end_idx / 31000) > self.pep_reset_index:
            # Delete existing peptide data to free memory
            del self.pep_seqs
            del self.pep_feats
            gc.collect()  # Run garbage collection to free up memory
            self.pep_seqs, self.pep_feats = self._read_peptides(self.pep_feat_root)
            self.pep_reset_index += 1

    def _fetch_latest_index(self, tcr_data_idx):
        pattern = f"affinity_tables_tcr_data_{tcr_data_idx}_(\d{{8}}_\d{{6}})"
        data_dirs = [m.group(0) for dir_name in os.listdir("./") if (m := re.match(pattern, dir_name))]

        if data_dirs:
            return max((max((int(os.path.splitext(x)[0].split("_")[2]) for x in os.listdir(data_dir)), default=0) for data_dir in data_dirs), default=0)

        return 0


def check_sampled_data_sanity(outdir, pkl_path, tcr_model_path, pep_model_path, model_paths, model_mode='softmax', pmhc_maxlen=18, device='cuda', pep_data=None):
    def _init_model(model_paths):
        # Use list comprehension for conciseness and filter paths with start condition
        return [_load_and_prepare_model(path) for path in model_paths if path.startswith("tabr_bert_fork")]

    def _load_and_prepare_model(model_path):
        # Initialize and load the model only once in a separate method for clarity
        model = nn.DataParallel(tcr_pmhc(mode=model_mode, pmhc_maxlen=pmhc_maxlen))
        model.load_state_dict(torch.load(model_path, map_location=device))
        if device == 'cuda':  # GPU
            model.cuda()
        model.eval()
        return model

    def _init_tcr_model(tcr_model_path):
        tcr_model = tcr_net()
        tcr_model = nn.DataParallel(tcr_model)
        tcr_model.load_state_dict(torch.load(tcr_model_path))
        tcr_model.cuda()
        tcr_model.eval()
        return tcr_model

    def _init_pep_model(pep_model_path, pmhc_maxlen):
        pmhc_model = pmhc_net(maxlen=pmhc_maxlen)
        pmhc_model = nn.DataParallel(pmhc_model)
        pmhc_model.load_state_dict(torch.load(pep_model_path))
        pmhc_model.cuda()
        pmhc_model.eval()
        return pmhc_model

    # Read data
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Initialize model
    models = _init_model(model_paths)
    tcr_model = _init_tcr_model(tcr_model_path)
    pep_model = _init_pep_model(pep_model_path, pmhc_maxlen)

    # Optionally consider random peptides
    if pep_data is not None:
        df_pep = pd.read_csv(pep_data)
        peps_random = df_pep.sample(n=100)['peptide']

    # Create/clear the CSV file with headers at the start
    Path(outdir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=['TCR', 'Peptide', 'Prediction']).to_csv(f"{outdir}/predictions.csv", index=False)

    # Run inference
    softmax = nn.Softmax()
    for tcr, peps in tqdm(data.items()):
        # Create temporary list for current TCR results
        current_results = []

        if pep_data is not None:
            peps = peps + peps_random.tolist()
        pep_loader = peptide_make_data(peps)
        tcr_loader = tcr_make_data([tcr] * len(peps))
        preds = []

        for tcr_tok, pep_tok in zip(tcr_loader, pep_loader):
            tcr_feat = tcr_model(tcr_tok[0])
            _, pep_feat = pep_model(pep_tok[0], pep_tok[1])
            pred_singles = []

            with torch.no_grad():
                for model in models:
                    logits = model(torch.cat([tcr_feat, pep_feat], dim=1))
                    pred_singles.append(softmax(logits)[:, 1])
                preds.append(torch.mean(torch.stack(pred_singles), dim=0))

        preds = torch.cat(preds)

        # Calculate medians
        first_median = torch.median(preds[:256]).item()
        remaining_median = torch.median(preds[256:]).item() if len(preds) > 256 else None

        # Print TCR information
        print(f"TCR: {tcr}, First 256 median: {first_median:.3f}, " +
              (f"Remaining median: {remaining_median:.3f}" if remaining_median is not None else "No remaining values"))

        # Store results for current TCR
        for pep, pred in zip(peps, preds):
            current_results.append({
                'TCR': tcr,
                'Peptide': pep,
                'Prediction': f"{pred:.4f}"
            })

        # Save current TCR results by appending to CSV
        pd.DataFrame(current_results).to_csv(f"{outdir}/predictions.csv",
                                           mode='a',
                                           header=False,
                                           index=False)

        # Save plot
        plot_scatter_with_gradients(preds, f"{outdir}/{tcr}.pdf")


def plot_scatter_with_gradients(preds, outfile):
    from matplotlib.colors import LinearSegmentedColormap
    preds_np = preds.cpu().numpy()
    n = len(preds_np)

    # Generate gradient colors from red to blue for the first 256 values
    red_to_blue = LinearSegmentedColormap.from_list("RedToBlue", ["red", "blue"], N=256)
    gradient_colors = [red_to_blue(i / 256) for i in range(256)]

    # Create a list of colors for all preds
    colors = ['black'] * n
    for i in range(min(256, n)):
        colors[i] = gradient_colors[i]

    # Create the scatter plot
    plt.figure(figsize=(10, 6))

    # Plot pseudo-labeled data
    plt.scatter(range(256), preds_np[:256], c=colors[:256], s=10, edgecolor='none')

    # Plot arbitrary peptides with a gap
    if n > 256:
        plt.scatter(np.arange(256, n) + 10, preds_np[256:], c=colors[256:], s=10, edgecolor='none')

    # plt.title('Predictions Scatter Plot with Gradients', fontsize=16)
    plt.ylabel('Predicted Affinity', fontsize=12)

    # Remove default x-axis ticks
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Add custom x-axis labels
    plt.text(128, plt.ylim()[0], 'Pseudo-labeled data', ha='center', va='top', fontsize=12)
    if n > 256:
        plt.text((256 + n) / 2 + 5, plt.ylim()[0], 'Arbitrary peptides', ha='center', va='top', fontsize=12)

    # Add a vertical line to separate the two datasets
    if n > 256:
        plt.axvline(x=256 + 5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(outfile, format='pdf')
    print(f"Plot saved as {outfile}")
