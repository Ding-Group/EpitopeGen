# Standard library imports
import itertools
import os
import time
import pickle
import random
from multiprocessing import Pool
from pathlib import Path

# Third-party library imports
import Levenshtein
import numpy as np
import pandas as pd
from Bio import Align, pairwise2
from Bio.Align import substitution_matrices
from tqdm import tqdm


class RandomGenerator:
    def __init__(self, train_csv, k=5, outdir="predictions", use_mhc=False):
        self.outdir = outdir
        Path(self.outdir).mkdir(parents=True, exist_ok=True)
        self.trainset = pd.read_csv(train_csv)
        self.k = k
        self.use_mhc = use_mhc
        self.length_dist = self._calculate_length_distribution()

    def _calculate_length_distribution(self):
        self.trainset = self.trainset.rename(columns={'text': 'tcr', 'label': 'epitope'})
        length_counts = self.trainset['epitope'].apply(len).value_counts(normalize=True)
        return length_counts.sort_index().cumsum()

    def _predict_single(self):
        p = random.random()
        for length, cum_prob in self.length_dist.items():
            if p < cum_prob:
                return ''.join(random.choices('ACDEFGHIKLMNPQRSTVWY', k=length))
        return ''

    def predict_all(self, test_set):
        df = pd.read_csv(test_set)
        df = df.rename(columns={'text': 'tcr', 'label': 'epitope'})
        predictions = [[row['tcr'], row['epitope']] + [self._predict_single() for _ in range(self.k)] for _, row in df.iterrows()]

        if self.use_mhc:
            cols_out = ['tcr', 'mhc', 'epitope'] + [f"pred_{i}" for i in range(self.k)]
        else:
            cols_out = ['tcr', 'epitope'] + [f"pred_{i}" for i in range(self.k)]

        df_result = pd.DataFrame(predictions, columns=cols_out)
        name = str(Path(test_set).stem)
        df_result.to_csv(f"{self.outdir}/random_{name}.csv", index=False)
        return df_result


class KNNSequenceGenerator:
    def __init__(self, train_csv, k=3, outdir="predictions", use_mhc=False):
        self.outdir = outdir
        self.use_mhc = use_mhc
        self.trainset = pd.read_csv(train_csv)
        self.k = k
        self.blosum62 = substitution_matrices.load("BLOSUM62")
        self.num_blosum_th = 500
        Path(self.outdir).mkdir(parents=True, exist_ok=True)

    def _blosum_distance(self, seq1, seq2):
        alignments = pairwise2.align.globalds(seq1, seq2, self.blosum62, -10, -0.5)
        return alignments[0][2] if alignments else 0

    def _levenshtein_distance(self, seq1, seq2):
        return Levenshtein.distance(seq1, seq2)

    def _find_most_similar_mhc(self, mhc):
        mhc_distances = self.trainset['mhc'].apply(lambda x: self._blosum_distance(mhc, x))
        min_mhc_dist = mhc_distances.min()
        return self.trainset[mhc_distances == min_mhc_dist]

    def _find_nearest_sequences(self, cdr3, df):
        if len(df) > self.num_blosum_th:
            df['levenshtein_dist'] = df['tcr'].apply(lambda tcr: self._levenshtein_distance(cdr3, tcr))
            candidate_df = df.nsmallest(self.num_blosum_th, 'levenshtein_dist')
        else:
            candidate_df = df

        candidate_df['dist_cdr3'] = candidate_df['tcr'].apply(lambda tcr: self._blosum_distance(cdr3, tcr))
        nearest_neighbors = candidate_df.nsmallest(self.k, 'dist_cdr3')
        return nearest_neighbors['epitope']

    def predict(self, args):
        """Generate an epitope sequence based on the nearest neighbors of the given cdr3 and mhc,
           focusing first on MHC similarity, then on minimal BLOSUM distance for TCRs."""
        cdr3, mhc, epi = (args[0], args[1], args[2]) if self.use_mhc else (args[0], args[1], None)
        try:
            if self.use_mhc:
                same_mhc = self.trainset[self.trainset['mhc'] == mhc]
                most_similar_mhc = same_mhc if not same_mhc.empty else self._find_most_similar_mhc(mhc)
                nearest_sequences = self._find_nearest_sequences(cdr3, most_similar_mhc)
            else:
                nearest_sequences = self._find_nearest_sequences(cdr3, self.trainset)
            return [args[0], args[1]] + nearest_sequences.tolist()
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None

    def predict_all(self, test_csv, n_proc=2):
        """Generate sequences for a DataFrame containing cdr3 and mhc columns using parallel processing."""
        print(f"KNN Sequence Generator: Start prediction.. n_proc={n_proc}")
        test_df = pd.read_csv(test_csv)
        test_df = test_df.rename(columns={'text': 'tcr', 'label': 'epitope'})
        cols_in = ['tcr', 'mhc', 'epitope'] if self.use_mhc else ['tcr', 'epitope']
        cols_out = ['tcr', 'mhc', 'epitope'] + [f'pred_{i}' for i in range(self.k)] if self.use_mhc else ['tcr', 'epitope'] + [f'pred_{i}' for i in range(self.k)]

        with Pool(n_proc) as p:
            preds = list(tqdm(p.imap(self.predict, test_df[cols_in].values), total=test_df.shape[0]))

        result_df = pd.DataFrame(preds, columns=cols_out)
        name = str(Path(test_csv).stem)
        result_df.to_csv(f"{self.outdir}/knn_{name}.csv", index=False)
        return result_df
