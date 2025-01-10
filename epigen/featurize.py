# Standard library imports
import argparse
import collections
import itertools
import logging
import multiprocessing as mp
import os
import pickle
import random
import re
import time
from datetime import datetime as dt
from pathlib import Path
from typing import List

# Third-party imports
import h5py
import matplotlib.pyplot as plt
import mhcnames
import numpy as np
import pandas as pd
from scipy.stats import skew
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# External imports
from tabr_bert_fork.bert_pmhc import BERT as pmhc_net
from tabr_bert_fork.bert_tcr import BERT as tcr_net
from tabr_bert_fork.tcr_pmhc_model import *  # NOTE: consider explicit imports instead of *
from epigen.utils import adheres_to_vocab, index_to_aa


class EpitopeFeaturizer:
    """
    Featurizes the epitopes using TABR-BERT (SSL) models

    epitope_data: str
        path to train_pmhc.csv
    model_path: str
        pretrained TABR-BERT mhc model (.pt)
    """
    def __init__(self, epitope_data, model_path, pseudo_sequence_file, outdir="pmhc_features", use_mhc=False):
        self.pseudo_sequence_file = pseudo_sequence_file
        self.use_mhc = use_mhc
        if self.use_mhc:
            self.pmhc_maxlen = 54
            self.alleles, self.epitopes = self._init_epitopes(epitope_data)
        else:
            self.pmhc_maxlen = 18
            self.epitopes = self._init_epitopes(epitope_data)
        self.model = self._init_model(model_path)
        self.dataloader = self._init_dataset(pseudo_sequence_file)
        self.outdir = outdir
        Path(outdir).mkdir(parents=True, exist_ok=True)

    def _init_epitopes(self, epitope_data):
        df = pd.read_csv(epitope_data)
        if self.use_mhc:
            return df['mhc'].tolist(), df['peptide'].tolist()
        else:
            return df['peptide'].tolist()

    def _init_model(self, model_path):
        pmhc_model = pmhc_net(maxlen=self.pmhc_maxlen)
        pmhc_model = nn.DataParallel(pmhc_model)
        pmhc_model.load_state_dict(torch.load(model_path))
        pmhc_model.cuda()
        pmhc_model.eval()
        return pmhc_model

    def _init_dataset(self, pseudo_sequence_file):
        if self.use_mhc:
            pmhc_loader = pmhc_make_data(self.alleles, self.epitopes)
        else:
            pmhc_loader = peptide_make_data(self.epitopes)
        return pmhc_loader

    def featurize_epitopes(self, partition_size=300000):
        def _save_data(part_cnt, features, outfile):
            # Modularized save logic for clarity
            outfile_path = os.path.join(self.outdir, f"{os.path.splitext(outfile)[0]}_{part_cnt}.pkl")
            with open(outfile_path, "wb") as f:
                pickle.dump(features, f)
            print(f"Saved: {outfile_path}")

        part_cnt = 0
        features_pep = {}
        features_mhc = {}
        total_feature_size = 0
        for batch in tqdm(self.dataloader, desc="Featurizing epitopes"):
            pmhc, seg_info = batch[0], batch[1]
            with torch.no_grad():
                _, feat = self.model(pmhc, seg_info)
                feat = feat.detach().cpu()

            for i, peptide_tensor in enumerate(pmhc):
                peptide_seq = "".join(index_to_aa[x.item()] for x in peptide_tensor[seg_info[i] == 1])
                if self.use_mhc:
                    mhc_seq = "".join(index_to_aa[x.item()] for x in peptide_tensor[seg_info[i] == 0])
                    features_pep[f"{peptide_seq}_{mhc_seq}"] = feat[i][seg_info[i] == 1].numpy()
                    features_mhc[f"{peptide_seq}_{mhc_seq}"] = feat[i][seg_info[i] == 0].numpy()
                else:
                    features_pep[f"{peptide_seq}"] = feat[i][seg_info[i] == 1].numpy()
            total_feature_size += pmhc.shape[0]

            if total_feature_size > partition_size:
                _save_data(part_cnt, features_pep, outfile='epi_features.pkl')
                if self.use_mhc:
                    _save_data(part_cnt, features_mhc, outfile='mhc_features.pkl')
                features_pep = {}
                features_mhc = {}
                part_cnt += 1
                total_feature_size = 0
        if features_pep:
            _save_data(part_cnt, features_pep, outfile='epi_features.pkl')
            if self.use_mhc:
                _save_data(part_cnt, features_mhc, outfile='mhc_features.pkl')


class VDJdbEpitopeFeaturizer(EpitopeFeaturizer):
    def __init__(self, epitope_data, model_path, pseudo_sequence_file, outdir="vdjdb_epi_features"):
        super().__init__(epitope_data, model_path, pseudo_sequence_file, outdir=outdir)

    def _init_epitopes(self, epitope_data):
        # Read epitope data from the csv file
        df = pd.read_csv(epitope_data)

        # Select rows with species as 'HomoSapiens' and vdjdb.score > 0
        selected_rows = df[(df['species'] == 'HomoSapiens') & (df['vdjdb.score'] > 0) & (df['mhc.class'] == 'MHCI')]
        print(f"VDJdb: total {len(selected_rows)} entries were selected ('HomoSapiens' and vdjdb.score > 0)")

        # Extract alleles and epitopes
        alleles = selected_rows['mhc.a'].tolist()
        epitopes = selected_rows['antigen.epitope'].tolist()

        # Map alleles to sequences
        allele_dict = pd.read_csv(self.pseudo_sequence_file)
        allele_dict = allele_dict.set_index("allele")

        alleles_seq = []
        for allele in alleles:
            try:
                allele_norm = mhcnames.normalize_allele_name(allele)
                a_seq = allele_dict.at[allele_norm, "sequence"]
            except:
                a_seq = "YFAMYGEKVAHTHVDTLYGVRYDHYYTWAVLAYTWYA"  # HLA-A*02:01
            alleles_seq.append(a_seq)
        return alleles_seq, epitopes


class TCRFeaturizer:
    """
    Featurizes the tcrs using TABR-BERT (SSL) models

    tcr_data: str
        dir to the standardized csv dataset
    model_path: str
        pretrained TABR-BERT tcr model (.pt)
    """
    def __init__(self, tcr_data, model_path, outdir="tcr_features"):
        self.tcrs = self._init_tcrs(tcr_data)
        self.model = self._init_model(model_path)
        self.dataloader = self._init_dataset()
        self.outdir = outdir
        Path(outdir).mkdir(parents=True, exist_ok=True)

    def _init_tcrs(self, tcr_data):
        df = pd.read_csv(tcr_data)
        selected_rows = df[(df['species'] == 'HomoSapiens') & (df['vdjdb.score'] > 0) & (df['mhc.class'] == 'MHCI')]
        return selected_rows['cdr3'].tolist()

    def _init_model(self, model_path):
        tcr_model = tcr_net()
        tcr_model = nn.DataParallel(tcr_model)
        tcr_model.load_state_dict(torch.load(model_path))
        tcr_model.cuda()
        tcr_model.eval()
        return tcr_model

    def _init_dataset(self):
        # Loading and processing data
        tcr_loader = tcr_make_data(self.tcrs)
        return tcr_loader

    def featurize_tcrs(self, outfile='tcr_features.pkl', bsz=16, n_proc=1, partition_size=300000):
        def _save_data(part_cnt, features, outfile):
            # Modularized save logic for clarity
            outfile_path = os.path.join(self.outdir, f"{os.path.splitext(outfile)[0]}_{part_cnt}.pkl")
            with open(outfile_path, "wb") as f:
                pickle.dump(features, f)
            print(f"Saved: {outfile_path}")

        part_cnt = 0
        features = {}
        total_feature_size = 0
        for batch in tqdm(self.dataloader, desc="Featurizing tcrs"):
            tcrs = batch[0]
            with torch.no_grad():
                feat = self.model(tcrs)
                feat = feat.detach().cpu()

            for i, tcr in enumerate(tcrs):
                tcr_seq = "".join([index_to_aa[x.item()] for x in tcr])
                features[tcr_seq] = feat[i].numpy()
            total_feature_size += tcrs.shape[0]

            if total_feature_size > partition_size:
                _save_data(part_cnt, features, outfile)
                features = {}
                part_cnt += 1
                total_feature_size = 0
        if features:
            _save_data(part_cnt, features, outfile)


class TCRDBFeaturizer(TCRFeaturizer):
    def __init__(self, tcr_data, model_path, outdir="tcr_features"):
        super().__init__(tcr_data, model_path, outdir)

    def _init_tcrs(self, tcr_data):
        return self.load_tcrdb(tcr_data)

    @staticmethod
    def load_tcrdb(tcr_data_path):
        df = pd.read_csv(tcr_data_path)
        return df['tcr'].tolist()

def _tcrdb_df_to_entries(fname: str) -> List[tuple]:
    """Helper function for processing TCRdb tables"""

    def tra_trb_from_str(s: str) -> str:
        if s.startswith("TRA"):
            return "TRA"
        elif s.startswith("TRB"):
            return "TRB"
        return "UNK"

    def infer_row_tra_trb(row) -> str:
        """Takes in a row from itertuples and return inferred TRA/TRB"""
        infers = []
        if "Vregion" in row._fields:
            infers.append(tra_trb_from_str(row.Vregion))
        if "Dregion" in row._fields:
            infers.append(tra_trb_from_str(row.Dregion))
        if "Jregion" in row._fields:
            infers.append(tra_trb_from_str(row.Jregion))
        if len(infers) == 0:
            return "UNK"
        # Use majority voting
        cnt = collections.Counter(infers)
        consensus, consensus_prop = cnt.most_common(1).pop()
        if consensus_prop / len(infers) > 0.5:
            return consensus
        return "UNK"  # No majority

    acc = os.path.basename(fname).split(".")[0]
    df = pd.read_csv(fname, delimiter="\t")
    entries = [
        (acc, row.RunId, row.AASeq, row.cloneFraction, infer_row_tra_trb(row))
        for row in df.itertuples(index=False)
    ]
    return entries
