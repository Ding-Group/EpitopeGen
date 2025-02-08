# Standard library imports
import os
import math
import itertools
import subprocess
import multiprocessing
from collections import Counter
from pathlib import Path
from itertools import combinations

# Third-party library imports
import Levenshtein
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from Bio import pairwise2
from Bio.Align import substitution_matrices
from Bio.Seq import Seq
from Bio.SeqUtils import ProtParam
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from matplotlib.collections import PolyCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from scipy.stats import gaussian_kde, mannwhitneyu, pearsonr
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from statannotations.Annotator import Annotator
from tqdm import tqdm
from umap import UMAP

# Project-specific imports
from bert_pmhc import BERT as pmhc_net
from bert_tcr import BERT as tcr_net
from tcr_pmhc_model import *


tcr_maxlen = 30
pmhc_maxlen = 54


class Evaluator:
    """
    Base class for Evaluators. To validate the generated epitopes
    in various aspects, read in a csv that contains GT and pred from
    many models. eval() method creates new columns per each model
    and they are saved, printed, and plotted.
    """
    def __init__(self, pred_csv, outdir):
        self.pred_csv = pred_csv
        self.outdir = outdir
        self.data = pd.read_csv(pred_csv)
        self.data['epitope'] = self.data['epitope'].fillna('VMPPRTLLL')  # Fill empty epitope with a random one
        # Corrected to ignore first three columns and include only model prediction columns
        self.model_columns = self.data.columns[2:]
        Path(outdir).mkdir(parents=True, exist_ok=True)

    def eval(self):
        raise NotImplementedError

    def _print_summary(self):
        # print summary stats along each newly created columns
        new_cols = self.data.columns[3 + len(self.model_columns):]

        # Use pandas' describe() to get various stats such as avg, std, min, max, and quartiles
        summary_stats = self.data[new_cols].describe()

        # Pretty print the summary statistics
        print("Summary Statistics for Each Model's Predictions:\n")
        print(summary_stats.to_string())

    def _plot_dist(self, desc=""):
        # Assuming `self.outdir` is the output directory attribute of the class
        new_cols = self.data.columns[3 + len(self.model_columns):]
        for col in new_cols:
            try:
                plt.figure(figsize=(10, 6))
                sns.histplot(self.data[col], kde=True, color='skyblue', label=f'{col}')
                plt.title(f'Distribution of Scores for {col}')
                plt.xlabel('Score')
                plt.ylabel('Frequency')
                plt.legend()

                # Create the output directory if it doesn't exist
                Path(self.outdir).mkdir(parents=True, exist_ok=True)
                # Save the plot in the specified 'outdir' directory with a descriptive filename
                plt.savefig(Path(self.outdir) / f'{desc}_{col}_distribution.pdf', format='pdf', dpi=300, bbox_inches='tight')
                plt.close()
            except:
                continue

    def _save(self, desc=""):
        output_file = f"{self.outdir}/evaluated_data_{desc}.csv"
        self.data.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")


class AffinityEvaluator:
    """
    For a single TCR and predictions for it, evaluate the alignment between them.
    Assume only the first row of pred_csv is that information and other subsequent rows are randomly matched peptides for comparison.
    """
    def __init__(self, pred_csvs, pmhc_data, outdir, num_controls=100, topk_values=[1, 5], pmhc_weight='', tcr_weight='', model_weights='',
                 cache_dir="__pycache__/AffinityEvaluator", device='cuda', use_cache=False, critique_pmhc_type="peptide"):
        self.pred_csvs = pred_csvs
        self.pred_csv_names = [str(Path(x).stem) for x in self.pred_csvs]
        self.pmhc_data = pd.read_csv(pmhc_data)
        self.outdir = outdir
        Path(self.outdir).mkdir(parents=True, exist_ok=True)
        self.data = [pd.read_csv(pred_csv).fillna({'epitope': 'VMPPRTLLL'}) for pred_csv in pred_csvs]
        self.cache_dir = cache_dir
        self.device = device

        self.num_controls = num_controls
        self.topk_values = topk_values  # What prediction to use in eval
        self.n_control_values = [10, 100]  # How many peptide controls to consider in eval

        self.pmhc_weight = pmhc_weight
        self.tcr_weight = tcr_weight
        self.model_weights = model_weights
        self.use_cache = use_cache
        self.softmax = nn.Softmax(dim=1)
        self.critique_pmhc_type = critique_pmhc_type
        self.pmhc_maxlen = 54 if critique_pmhc_type == "pmhc" else 18

        self._init_cache()
        self._init_models()

    def _init_cache(self):
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.cache_dir}/features").mkdir(parents=True, exist_ok=True)

    def _init_models(self):
        def load_model(model_class, weight_path):
            model = nn.DataParallel(model_class)
            model.load_state_dict(torch.load(weight_path, map_location=self.device))
            if self.device == 'cuda':
                model.cuda()
            model.eval()
            return model

        print("Loading pMHC model, TCR, and TCR-pMHC models..")
        self.pmhc_model = load_model(pmhc_net(maxlen=self.pmhc_maxlen), self.pmhc_weight)
        self.tcr_model = load_model(tcr_net(maxlen=30), self.tcr_weight)

        self.models = [
            load_model(tcr_pmhc(mode="softmax", pmhc_maxlen=self.pmhc_maxlen), weight)
            for weight in self.model_weights
        ]
        print("Models were successfully loaded!")

    def _prepare_tcr_features(self, df):
        tcr_cache_path = f"{self.cache_dir}/features/tcr_feat.pt"

        if os.path.exists(tcr_cache_path) and self.use_cache:
            print(f"Load from TCR cache: {tcr_cache_path}")
            self.tcr_feat = torch.load(tcr_cache_path)
        else:
            # print("Featurizing TCRs using BERT..")
            tcr_loader = tcr_make_data(df['tcr'].tolist())
            tcr_features = []

            with torch.no_grad():
                for batch in tcr_loader:
                    feat = self.tcr_model(batch[0]).detach().cpu()
                    tcr_features.append(feat)

            self.tcr_feat = torch.cat(tcr_features, dim=0)
            Path(os.path.dirname(tcr_cache_path)).mkdir(parents=True, exist_ok=True)
            torch.save(self.tcr_feat, tcr_cache_path)

    def _prepare_pmhc_features(self, df):
        pmhc_cache_path = os.path.join(self.cache_dir, f"features/{self.critique_pmhc_type}_feat.pt")

        def get_pmhc_loader():
            if self.critique_pmhc_type == "pmhc":
                from tcr_pmhc_model import pmhc_make_data
                return pmhc_make_data(df['mhc'].tolist(), df['epitope'].tolist())
            return peptide_make_data(df['epitope'].tolist())

        def featurize_pmhc(pmhc_loader):
            pmhc_features = []

            with torch.no_grad():
                for batch in pmhc_loader:
                    pmhc, seg_info = batch[0], batch[1]
                    _, feat = self.pmhc_model(pmhc, seg_info)
                    pmhc_features.append(feat.detach().cpu())

            return torch.cat(pmhc_features, dim=0)

        def save_pmhc_cache(features, cache_path):
            Path(os.path.dirname(cache_path)).mkdir(parents=True, exist_ok=True)
            torch.save(features, cache_path)

        if os.path.exists(pmhc_cache_path) and self.use_cache:
            print(f"Load from cache: {pmhc_cache_path}")
            self.pmhc_feat = torch.load(pmhc_cache_path)
        else:
            # print(f"Featurize {self.critique_pmhc_type} with predicted epitopes using BERT..")
            pmhc_loader = get_pmhc_loader()
            self.pmhc_feat = featurize_pmhc(pmhc_loader)
            save_pmhc_cache(self.pmhc_feat, pmhc_cache_path)

    def _prepare_features(self, df):
        self._prepare_tcr_features(df)
        self._prepare_pmhc_features(df)

    def _calc_metric(self, df):
        generated_affinity = df['Average_Affinity'].iloc[0]  # first row of 'Average_Affinity'
        random_affinities = df['Average_Affinity'].iloc[1:]  # all but the first row of 'Average_Affinity'

        metrics = []

        for n in self.n_control_values:
            sampled_affinities = random_affinities.iloc[:n]
            mean_rand_aff = sampled_affinities.mean()
            std_rand_aff = sampled_affinities.std()

            mu, std = stats.norm.fit(sampled_affinities)
            cdf_value = stats.norm.cdf(generated_affinity, loc=mu, scale=std)
            percentile_rank = stats.percentileofscore(sampled_affinities, generated_affinity)

            # Calculate the 95% confidence interval for the mean of sampled affinities
            se = std_rand_aff / np.sqrt(len(sampled_affinities))
            ci_upper = stats.t.interval(0.95, len(sampled_affinities) - 1, loc=mean_rand_aff, scale=se)[1]

            metrics.append((n, cdf_value, percentile_rank, generated_affinity, mean_rand_aff, std_rand_aff, ci_upper - mean_rand_aff))

        return metrics

    def eval_df(self, df):
        self._prepare_features(df)
        # For the GPT2-predicted epitopes
        x = torch.cat([self.tcr_feat, self.pmhc_feat], dim=1)
        with torch.no_grad():
            x = x.to(self.device)
            for i, model in enumerate(self.models):
                logits = model(x)
                aff = self.softmax(logits)[:, 1]
                df[f'aff_{i}'] = aff.cpu()
        df['Average_Affinity'] = df[['aff_0','aff_1','aff_2','aff_3','aff_4']].mean(axis=1)

        metrics = self._calc_metric(df)
        return metrics

    def _sample_neg_controls(self, mode='rand_pep_out'):
        return (self.pmhc_data[['allele', 'peptide']].sample(n=self.num_controls).rename(columns={'allele': 'mhc'})
                if mode == 'rand_pmhc_out' else self.pmhc_data[['peptide']].sample(n=self.num_controls)).reset_index(drop=True)

    def _create_df_with_control_peps(self, row, df_neg, topk):
        tcr_value = row['tcr']
        first_row_data = {'tcr': [tcr_value], 'epitope': [row[f'pred_{topk-1}']]}  # topk - 1 because the prediction file index starts from pred_0 but I want to input func args from topk: 1, 5, 10, 20 for convenience
        first_row = pd.DataFrame(first_row_data)
        df_neg = df_neg.rename(columns={'peptide': 'epitope'})
        final_df = pd.concat([first_row, pd.concat([pd.DataFrame({'tcr': [tcr_value] * len(df_neg)}), df_neg], axis=1)]).reset_index(drop=True)
        return final_df.rename(columns={'peptide': 'epitope'})

    def eval(self):
        metrics = {k: {pred_csv: [] for pred_csv in self.pred_csv_names} for k in self.topk_values}
        for topk in self.topk_values:
            print(f"Evaluate for topk={topk}..")
            for tcr_idx in tqdm(range(min([len(x) for x in self.data]))):  # rows in the test set
                df_neg = self._sample_neg_controls()
                for pred_csv, data in zip(self.pred_csv_names, self.data):  # For each baseline
                    df = self._create_df_with_control_peps(data.iloc[tcr_idx], df_neg, topk)
                    metrics[topk][pred_csv].append(self.eval_df(df))

        with open(f"{self.outdir}/metrics.pkl", "wb") as f:
            pickle.dump(metrics, f)
        print(f"{self.outdir}/metrics.pkl was saved. ")
        parameters = {
            "topk_values": self.topk_values,
            "pred_csv_names": self.pred_csv_names,
            "n_control_values": self.n_control_values
        }
        with open(f"{self.outdir}/parameters.pkl", "wb") as f:
            pickle.dump(parameters, f)
        print(f"{self.outdir}/parameters.pkl was saved. ")

        self.draw_line_plots(metrics)
        self.draw_violin_plots(metrics)
        # self.draw_scatter_plots(metrics)
        return metrics

    def draw_line_plots(self, metrics):
        # Summarize the results and plot
        summary_data = []
        for topk in self.topk_values:
            for pred_csv in self.pred_csv_names:
                mets = metrics[topk][pred_csv]
                for n in self.n_control_values:
                    percentile_ranks = [m[2] for sublist in mets for m in sublist if m[0] == n]
                    for pr in percentile_ranks:
                        summary_data.append((topk, pred_csv, n, pr))
        df_summary = pd.DataFrame(summary_data, columns=['TopK', 'Prediction', 'NControls', 'PercentileRank'])
        df_summary.to_csv(f"{self.outdir}/df_summary.csv", index=False)
        print(f'{self.outdir}/df_summary.csv')

        # Plot: Percentile Rank vs Number of Controls
        plt.figure(figsize=(16, 10))  # Increased figure size to accommodate the legend
        sns.lineplot(data=df_summary, x='NControls', y='PercentileRank', hue='Prediction', style='TopK', markers=True, dashes=False)
        plt.title('Percentile Rank vs Number of Controls', fontsize=16)
        plt.xlabel('Number of Controls', fontsize=16)
        plt.ylabel('Percentile Rank', fontsize=16)
        # Move legend outside the plot
        plt.legend(title='Prediction/TopK', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        # plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.outdir}/lineplot_controls.pdf', dpi=300, bbox_inches='tight', format='pdf')
        plt.close()

        # Plot: Percentile Rank vs TopK (Fixed Number of Controls)
        fixed_n = self.n_control_values[-1]
        fixed_data = []
        for topk in self.topk_values:
            for pred_csv in self.pred_csv_names:
                mets = metrics[topk][pred_csv]
                percentile_ranks = [m[2] for sublist in mets for m in sublist if m[0] == fixed_n]
                for pr in percentile_ranks:
                    fixed_data.append((topk, pred_csv, pr))
        df_fixed = pd.DataFrame(fixed_data, columns=['TopK', 'Prediction', 'PercentileRank'])

        plt.figure(figsize=(16, 10))  # Increased figure size
        sns.lineplot(data=df_fixed, x='TopK', y='PercentileRank', hue='Prediction', markers=True, dashes=False)
        # plt.title(f'Percentile Rank vs TopK (Fixed NControls = {fixed_n})', fontsize=16)
        plt.xlabel('TopK', fontsize=16)
        plt.ylabel('Percentile Rank', fontsize=16)
        # Move legend outside the plot
        plt.legend(title='Prediction', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        # plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.outdir}/lineplot_topk.pdf', dpi=300, bbox_inches='tight', format='pdf')
        plt.close()

    def draw_violin_plots(self, metrics):
        fixed_n = self.n_control_values[-1]
        for topk in self.topk_values:
            affinity_data = []
            rank_data = []
            for pred_csv in self.pred_csv_names:
                mets = metrics[topk][pred_csv]
                for sublist in mets:
                    for m in sublist:
                        if m[0] == fixed_n:
                            affinity_data.append((pred_csv, m[3], 'Generated Affinity'))
                            affinity_data.append((pred_csv, m[4], 'Random Affinity'))
                            rank_data.append((pred_csv, m[2]))  # Append percentile rank

            df_affinity = pd.DataFrame(affinity_data, columns=['Prediction', 'Affinity', 'Type'])
            df_rank = pd.DataFrame(rank_data, columns=['Prediction', 'Percentile Rank'])

            # Set up the plot
            plt.figure(figsize=(10, 8))
            # plt.title(f'Affinity Prediction Values for TopK = {topk} and Fixed N = {fixed_n}', fontsize=20)

            # Color palette
            palette = sns.color_palette("husl", n_colors=len(self.pred_csv_names))

            # Plot the distribution of affinity values for each prediction
            for i, pred_csv in enumerate(self.pred_csv_names):
                sns.histplot(data=df_affinity[(df_affinity['Type'] == 'Generated Affinity') & (df_affinity['Prediction'] == pred_csv)],
                             x='Affinity', label=pred_csv, kde=False, color=palette[i], bins=30, alpha=0.5)

            # Customize the plot
            plt.xlabel('Affinity', fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            plt.legend(title='Prediction')
            # plt.grid(True, linestyle='--', linewidth=0.5)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            plt.tight_layout()
            plt.savefig(f'{self.outdir}/affinity_distribution_values_topk_{topk}.pdf', dpi=300, format='pdf')
            plt.close()

            # Define the color palette outside the loop (if not already defined)
            palette = sns.color_palette("husl", n_colors=len(self.pred_csv_names))

            # Violin plot for each topk - Percentile Rank
            plt.figure(figsize=(14, 8))
            vplot_rank = sns.violinplot(data=df_rank, x='Prediction', y='Percentile Rank',
                                        inner=None, cut=0, palette=palette)

            # Customize the violin plot appearance
            for artist in vplot_rank.findobj(PolyCollection):
                artist.set_edgecolor('black')
                artist.set_alpha(0.7)

            # Customize the median line
            for line in vplot_rank.lines:
                if line.get_linestyle() == '-':
                    line.set_color('black')
                    line.set_linewidth(2)

            # Add labels and title
            # plt.xlabel('Prediction', fontsize=16)
            plt.ylabel('Percentile Rank', fontsize=16)
            # plt.title(f'Percentile Rank Distribution for TopK = {topk} and Fixed N = {fixed_n}', fontsize=16)
            # plt.grid(True, linestyle='--', linewidth=0.5)
            plt.gca().tick_params(axis='both', which='major', labelsize=12)
            plt.tight_layout()
            plt.savefig(f'{self.outdir}/percentile_rank_distribution_topk_{topk}.pdf', dpi=300, format='pdf')
            plt.close()

    def draw_scatter_plots(self, metrics):
        fixed_n = self.n_control_values[-1]
        # Assuming calculate_cohens_d is a function you've defined elsewhere
        def calculate_cohens_d(generated, random):
            return (np.mean(generated) - np.mean(random)) / np.sqrt((np.std(generated) ** 2 + np.std(random) ** 2) / 2)

        # Scatter Plot for each pred_csv and topk
        for topk in self.topk_values:
            for pred_csv in self.pred_csv_names:
                scatter_data = []
                mets = metrics[topk][pred_csv]
                for sublist in mets:
                    for m in sublist:
                        if m[0] == fixed_n:
                            scatter_data.append((m[3], m[4], m[6]))  # generated_affinity, mean_rand_aff, CI

                # Sample at most 1000 points from scatter_data
                if len(scatter_data) > 300:
                    scatter_data = random.sample(scatter_data, 300)

                # Unzip the data
                generated_affinities, mean_rand_affinities, CIs = zip(*scatter_data)
                cohens_d = calculate_cohens_d(generated_affinities, mean_rand_affinities)

                # Create scatter plot with main axes
                fig, main_ax = plt.subplots(figsize=(10, 8))

                # Use errorbar instead of scatter to include error bars
                # main_ax.errorbar(mean_rand_affinities, generated_affinities, xerr=CIs, fmt='o', label='EpiGen', alpha=0.7, capsize=3, color='navy')
                main_ax.scatter(mean_rand_affinities, generated_affinities, label='EpiGen', alpha=0.7, color='navy', s=5)

                # Add y=x line
                max_affinity = max(max(generated_affinities), max(mean_rand_affinities))
                main_ax.plot([0, max_affinity], [0, max_affinity], 'k--', label='y=x')

                # Add labels, title, and legend to the main plot
                main_ax.set_xlabel('Mean Affinity by Random Epitopes', fontsize=16)
                main_ax.set_ylabel('Affinity by Generated Epitope', fontsize=16)
                main_ax.legend()

                # Set grid, limits, and layout for the main plot
                # main_ax.grid(True)
                main_ax.set_xlim(0, max(0.2, max(mean_rand_affinities)))
                main_ax.set_ylim(0, max(generated_affinities))

                # Instead of using seaborn on the main axis, let's create a secondary axis for the KDE plot
                divider = make_axes_locatable(main_ax)
                right_ax = divider.append_axes("right", size=1.2, pad=0.1, sharey=main_ax)

                # Calculate the density of the generated affinities and plot it on the new axes
                sns.kdeplot(generated_affinities, ax=right_ax, vertical=True, color='royalblue', alpha=0.5)
                right_ax.set_ylim(main_ax.get_ylim())  # Match the y-axis limits with the main plot
                right_ax.yaxis.set_visible(False)  # Optionally hide the right y-axis
                # Remove the word 'Density' and the scale information from the density plot
                right_ax.set_xlabel('')
                right_ax.set_xticks([])

                plt.figtext(0.4, 0.85, f"Cohen's d: {cohens_d:.2f}", ha='center', va='center', fontsize=16, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.2'))

                # Save the plot
                plt.savefig(f'{self.outdir}/scatter_plot_{pred_csv}_topk_{topk}.pdf', dpi=300, format='pdf')


def calculate_cohens_d(group1, group2):
    # Calculate the mean and standard deviation of each group
    mean1, mean2 = np.mean(group1), np.mean(group2)
    sd1, sd2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # Calculate the pooled standard deviation
    n1, n2 = len(group1), len(group2)
    sd_pooled = np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))

    # Calculate Cohen's d
    d = (mean1 - mean2) / sd_pooled
    return d


class ChemicoEvaluator(Evaluator):
    """
    Get the chemical properties of the generated epitopes using Biopython's ProtParam.
    Documentation:
    - ProtParam: https://web.expasy.org/cgi-bin/protparam/protparam
    - Bio.SeqUtils.ProtParam: https://biopython.org/wiki/ProtParam
    """
    def __init__(self, pred_csv, outdir, cache_dir="__pycache__/ChemicoEvaluator"):
        super().__init__(pred_csv, f"{outdir}/Chem")
        Path(outdir).mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _analyze(self, sequence):
        pa = ProtParam.ProteinAnalysis(str(sequence))
        aa_perc = pa.get_amino_acids_percent()
        mw = pa.molecular_weight()
        am = pa.aromaticity()
        ii = pa.instability_index()
        ip = pa.isoelectric_point()
        sec_struct = pa.secondary_structure_fraction()[2]  # [helix, turn, sheet]
        eps_prot, _ = pa.molar_extinction_coefficient()  # [reduced, oxidized]
        return {"aa_perc": aa_perc, "mw": mw, "am": am, "ii": ii, "ip": ip,
                "sec_struct": sec_struct, "eps_prot": eps_prot}

    def eval(self):
        desc = str(Path(self.pred_csv).stem)
        for idx, model in enumerate(['epitope'] + self.model_columns.tolist()):
            if idx == 3:
                break
            metrics_list = [self._analyze(epitope) for epitope in self.data[model]]
            metrics_df = pd.DataFrame(metrics_list)
            metrics_df.columns = [f"{col}_{model}" for col in metrics_df.columns]

            # Concatenate the original data with the inferred data horizontally
            self.data = pd.concat([self.data, metrics_df], axis=1)
        self._print_summary()
        self._plot_dist(desc=f"Chem_{desc}")
        self._save(desc=f"Chem_{desc}")


def draw_dist_Chemico(pred_chem_csv, rand_chem_csv, outdir, label_col='epitope', pred_col='pred_0'):
    df_perfect = pd.read_csv(pred_chem_csv)
    df_random = pd.read_csv(rand_chem_csv)
    # Filter for unique rows based on the 'epitope' column
    df_perfect = df_perfect.drop_duplicates(subset='epitope', keep='first')
    df_random = df_random.drop_duplicates(subset='epitope', keep='first')

    # postfixes = ['epitope', 'label']
    postfixes = [pred_col, label_col]
    attributes = ['mw', 'am', 'ii', 'ip', 'sec_struct', 'eps_prot']
    attribute_map = {'mw': 'Molecular Weight', 'am': 'Aromacity', 'ii': 'Instability Index', 'ip': 'Isoelectric Point', 'sec_struct': 'Secondary Structure (sheet)', 'eps_prot': 'Extinction Coefficient'}

    # Set style for all plots
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)

    for attribute in attributes:
        # Preparing data for violin plot
        data_to_plot = []
        labels = []
        for postfix in postfixes:
            col_perfect = f"{attribute}_{postfix}"
            col_random = f"{attribute}_{postfix}"

            # Append perfect predictions data
            if col_perfect in df_perfect.columns:
                temp_df = df_perfect[[col_perfect]].copy()
                if postfix == 'pred_0':
                    temp_df['Type'] = 'EpiGen'
                elif postfix == 'epitope':
                    temp_df['Type'] = 'Natural'
                temp_df.rename(columns={col_perfect: 'Value'}, inplace=True)
                data_to_plot.append(temp_df)

            # Append random predictions data
            if col_random in df_random.columns and postfix != label_col:
                temp_df = df_random[[col_random]].copy()
                temp_df['Type'] = 'RandGen'
                temp_df.rename(columns={col_random: 'Value'}, inplace=True)
                data_to_plot.append(temp_df)

        final_df = pd.concat(data_to_plot, ignore_index=True)

        # Create tall and skinny figure
        plt.figure(figsize=(10, 10))

        # Define the desired order - putting EpiGen and Natural together
        plot_order = ['EpiGen', 'Natural', 'RandGen']

        # Create violin plot
        ax = sns.violinplot(x='Type', y='Value', data=final_df,
                            palette='Set3',
                            split=False,
                            inner='box',
                            linewidth=1.5,
                            order=plot_order)

        # Remove x-axis label
        ax.set_xlabel('')  # or ax.set_xlabel(None)

        # Perform statistical tests
        def compute_statistics(group1, group2):
            # Mann-Whitney U test
            stat, pval = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            # Additional statistics
            mean_diff = group1.mean() - group2.mean()
            median_diff = group1.median() - group2.median()
            cohen_d = (group1.mean() - group2.mean()) / np.sqrt((group1.var() + group2.var()) / 2)
            return {
                'pvalue': pval,
                'mean_diff': mean_diff,
                'median_diff': median_diff,
                'cohen_d': cohen_d,
                'group1_mean': group1.mean(),
                'group2_mean': group2.mean(),
                'group1_median': group1.median(),
                'group2_median': group2.median(),
                'group1_std': group1.std(),
                'group2_std': group2.std()
            }

        # Define pairs for comparison
        pairs = [('EpiGen', 'Natural'), ('EpiGen', 'RandGen'), ('Natural', 'RandGen')]

        # Calculate statistics
        stats_results = []
        p_values = []
        for pair in pairs:
            group1_data = final_df[final_df['Type'] == pair[0]]['Value']
            group2_data = final_df[final_df['Type'] == pair[1]]['Value']
            stats_dict = compute_statistics(group1_data, group2_data)
            stats_dict['comparison'] = f"{pair[0]} vs {pair[1]}"
            stats_results.append(stats_dict)
            p_values.append(stats_dict['pvalue'])

        # Adjust p-values for multiple comparisons
        import statsmodels
        adjusted_p_values = statsmodels.stats.multitest.multipletests(p_values, method='bonferroni')[1]

        # Add adjusted p-values to stats_results
        for stats_dict, adj_p in zip(stats_results, adjusted_p_values):
            stats_dict['adjusted_pvalue'] = adj_p

        # Save statistics to CSV
        stats_df = pd.DataFrame(stats_results)
        output_dir = Path(outdir) / 'Chem'
        stats_df.to_csv(output_dir / f'{attribute}_statistics.csv', index=False)

        # Add custom statistical annotations with more information
        def add_stat_annotations(ax, pairs, stats_results):
            y_max = final_df['Value'].max()
            y_min = final_df['Value'].min()
            y_range = y_max - y_min

            # Function to convert p-value to stars
            def p_to_stars(p):
                if p < 0.001:
                    return '***'
                elif p < 0.01:
                    return '**'
                elif p < 0.05:
                    return '*'
                else:
                    return 'ns'

            # Add annotations for each pair
            for i, (pair, stats_result) in enumerate(zip(pairs, stats_results)):
                x1 = plot_order.index(pair[0])
                x2 = plot_order.index(pair[1])

                # Calculate y position for the bar
                y = y_max + (i + 1) * y_range * 0.1

                # Draw the bracket
                line_height = y_range * 0.05
                plt.plot([x1, x1, x2, x2],
                        [y, y + line_height, y + line_height, y],
                        'k-', linewidth=1)

                # Add star annotation and p-value
                stars = p_to_stars(stats_result['adjusted_pvalue'])
                plt.text((x1 + x2) / 2, y + line_height,
                        f'{stars}\np={stats_result["adjusted_pvalue"]:.2e}',
                        ha='center', va='bottom')

            # Add median values on the violins
            # for i, type_name in enumerate(plot_order):
            #     median = final_df[final_df['Type'] == type_name]['Value'].median()
            #     plt.text(i, y_min, f'median:\n{median:.2f}',
            #             ha='center', va='bottom')

        # Add the annotations
        add_stat_annotations(ax, pairs, stats_results)

        # Customize the plot
        plt.ylabel(f'{attribute_map[attribute]}', fontsize=16)
        plt.xticks(rotation=0, ha='center', fontsize=16)
        plt.yticks(fontsize=16)
        sns.despine(trim=True, offset=10)
        plt.tight_layout()

        # Save the plot
        plt.savefig(output_dir / f'{attribute}_violin_distribution.pdf',
                    format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

        # Print summary to console
        print(f"\nStatistics for {attribute}:")
        for stats_result in stats_results:
            print(f"\n{stats_result['comparison']}:")
            print(f"Adjusted p-value: {stats_result['adjusted_pvalue']:.2e}")
            print(f"Cohen's d: {stats_result['cohen_d']:.3f}")
            print(f"Mean difference: {stats_result['mean_diff']:.3f}")
            print(f"Median difference: {stats_result['median_diff']:.3f}")


def blosum62_score(seq1, seq2):
        blosum = matlist.blosum62
        score = 0
        for a, b in zip(seq1, seq2):
            if (a, b) in blosum:
                score += blosum[(a, b)]
            elif (b, a) in blosum:
                score += blosum[(b, a)]
            else:
                score += blosum[('X', 'X')]  # Handle rare cases or non-standard amino acids
        return score


def eval_naturalness(pred_csv, blastp_result, outdir=None, rand_csv=None, blastp_result_rand=None):
    def BLOSUM(seq1, seq2):
        # Uses Biopython's pairwise2 module and BLOSUM62 matrix for scoring
        matrix = substitution_matrices.load('BLOSUM62')  # use matrix names to load
        alignments = pairwise2.align.globaldx(seq1, seq2, matrix)
        # Assuming the best alignment is the first one
        score = alignments[0][2]
        return score

    if outdir is None:
        outdir = str(Path(blastp_result).parent.parent)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    # Draw a plot where the x-axis is the BLOSUM distance and y-axis is the log E value
    df_pred = pd.read_csv(pred_csv)
    df_blastp = pd.read_csv(
        blastp_result,
        sep='\t',
        names=['pred_0', 'accession', 'match', 'match_len', 'remove', 'gap', 'epi_s', 'epi_e', 'prot_s', 'prot_e', 'evalue', 'score']
    )
    print(f"df_blastp was read. # of rows: {len(df_blastp)}")
    # Filter blastp result by min E-value
    # idx = df_blastp.groupby(['epitope'])['evalue'].idxmin()
    idx = df_blastp.groupby(['pred_0'])['evalue'].transform(np.min) == df_blastp['evalue']
    df_blastp_filtered = df_blastp.loc[idx]
    print(f"df_blastp_filtered by E-value: total {len(df_blastp_filtered)} rows")

    df = pd.merge(df_pred, df_blastp_filtered, on='pred_0', how='left').dropna(subset=['accession'])

    # Calculate BLOSUM62 scores for each pair
    print("Calculating BLOSUM distances..")
    df['blosum_score'] = df.apply(lambda row: BLOSUM(row['epitope'], row['pred_0']), axis=1)

    if rand_csv and blastp_result_rand:
        df_rand = pd.read_csv(rand_csv)
        df_blastp_rand = pd.read_csv(
            blastp_result_rand,
            sep='\t',
            names=['pred_0', 'accession', 'match', 'match_len', 'remove', 'gap', 'epi_s', 'epi_e', 'prot_s', 'prot_e', 'evalue', 'score']
        )
        print(f"df_blastp_rand was read. # of rows: {len(df_blastp)}")
        # Filter blastp result by min E-value
        # idx_rand = df_blastp_rand.groupby(['epitope'])['evalue'].idxmin()
        idx_rand = df_blastp_rand.groupby(['pred_0'])['evalue'].transform(min) == df_blastp_rand['evalue']
        df_blastp_filtered_rand = df_blastp_rand.loc[idx_rand]
        print(f"df_blastp_filtered_rand by E-value: total {len(df_blastp_filtered_rand)} rows")

        df_rand_merged = pd.merge(df_rand, df_blastp_filtered_rand, on='pred_0', how='left').dropna(subset=['accession'])

        # Calculate BLOSUM62 scores for each pair
        print("Calculating BLOSUM distances..")
        df_rand_merged['blosum_score'] = df_rand_merged.apply(lambda row: BLOSUM(row['epitope'], row['pred_0']), axis=1)

    # Function to add jitter
    def add_jitter(x, y, amount=1.0):
        return x + 3 * np.random.normal(0, amount * 4, size=len(x)), y + 2 * np.random.normal(0, amount, size=len(y))

    # Plotting
    plt.figure(figsize=(10, 6))

    # Add jitter to both datasets
    jitter_amount = 0.1  # Adjust this value to control the amount of jitter

    x_epi, y_epi = add_jitter(df['blosum_score'], np.log10(df['evalue']), jitter_amount)
    x_rand, y_rand = add_jitter(df_rand_merged['blosum_score'], np.log10(df_rand_merged['evalue']), jitter_amount)

    # Plot for df and df_rand_merged with jitter
    plt.scatter(x_epi, y_epi, color='blue', alpha=0.5, edgecolor='none', s=10, label='EpiGen')
    plt.scatter(x_rand, y_rand, color='red', alpha=0.5, edgecolor='none', s=10, label='RandGen')

    # Enhancing plot design
    plt.xlabel('BLOSUM62 Score', fontsize=14, fontweight='bold')
    plt.ylabel('Log10 E-value', fontsize=14, fontweight='bold')
    plt.title('Comparison of BLOSUM62 Score vs E-value', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Methods', title_fontsize='13', fontsize='12', loc='upper right')

    # Adjusting axes to a linear scale since log10 is manually applied
    plt.yscale('linear')  # Use linear scale because we've already transformed e-value with log10

    # Save the plot
    plt.savefig(f"{outdir}/naturalness.pdf", dpi=300)
    print(f"Plot saved at {outdir}/naturalness.pdf")

    # [2] Create histograms of log E values
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Calculate statistics first
    epigen_evalues = df['evalue']
    randgen_evalues = df_rand_merged['evalue']
    statistic, p_value = stats.mannwhitneyu(epigen_evalues, randgen_evalues, alternative='less')
    epigen_median = np.median(epigen_evalues)
    randgen_median = np.median(randgen_evalues)

    # Calculate common bins for both histograms
    min_val = min(np.log10(epigen_evalues).min(), np.log10(randgen_evalues).min())
    max_val = max(np.log10(epigen_evalues).max(), np.log10(randgen_evalues).max())
    bins = np.linspace(min_val, max_val, 50)

    # Plot histograms and get their maximum y-values
    n1, _, _ = ax1.hist(np.log10(df['evalue']), bins=bins, color='#7eb4db', alpha=0.7, label='EpiGen')
    n2, _, _ = ax2.hist(np.log10(df_rand_merged['evalue']), bins=bins, color='red', alpha=0.7, label='RandGen')

    # Find the maximum y value between both plots
    max_y = max(n1.max(), n2.max())

    # Set the same y limits for both plots
    ax1.set_ylim(0, max_y * 1.1)  # Adding 10% padding
    ax2.set_ylim(0, max_y * 1.1)

    # Customize first subplot
    ax1.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax1.legend(title='Method', title_fontsize='13', fontsize='12', loc='upper right')
    ax1.tick_params(labelsize=12)
    # Add median annotation for EpiGen
    ax1.axvline(np.log10(epigen_median), color='black', linestyle='--', alpha=0.5)
    ax1.text(0.02, 0.95, f'Median = {epigen_median:.2e}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top')

    # Customize second subplot
    ax2.set_xlabel('Log10 E-value', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax2.legend(title='Method', title_fontsize='13', fontsize='12', loc='upper right')
    ax2.tick_params(labelsize=12)
    # Add median annotation for RandGen
    ax2.axvline(np.log10(randgen_median), color='black', linestyle='--', alpha=0.5)
    ax2.text(0.02, 0.95, f'Median = {randgen_median:.2e}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top')

    # Add p-value annotation in the center between subplots
    fig.text(0.02, 0.5, f'Mann-Whitney U test\np-value = {p_value:.2e}',
             fontsize=10, verticalalignment='center')

    # Adjust the layout to prevent overlap
    plt.tight_layout()

    # Save the new histogram plot
    plt.savefig(f"{outdir}/evalue_histogram.pdf", dpi=300)
    print(f"Histogram plot saved at {outdir}/evalue_histogram.pdf")

    # Close the figure to free up memory
    plt.close()

    # Print the statistical results
    print(f"Mann-Whitney U test statistic: {statistic}")
    print(f"p-value: {p_value}")

    # Interpret the results
    alpha = 0.05  # Set your desired significance level
    if p_value < alpha:
        print("The e-values of the epitopes generated by EpiGen are significantly lower than those of the epitopes from RandGen.")
    else:
        print("There is not enough evidence to conclude that the e-values of the epitopes generated by EpiGen are significantly lower than those of the epitopes from RandGen.")

    print(f"Median e-value for EpiGen: {epigen_median}")
    print(f"Median e-value for RandGen: {randgen_median}")

    # [3] Create a histogram of BLOSUM scores
    plt.figure(figsize=(10, 6))

    # Plot the histogram
    plt.hist(df['blosum_score'], bins=50, alpha=0.5, color='blue', label='EpiGen')
    plt.hist(df_rand_merged['blosum_score'], bins=50, alpha=0.5, color='red', label='RandGen')

    # Enhancing plot design
    plt.xlabel('BLOSUM', fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=14, fontweight='bold')
    # plt.title('Distribution of BLOSUM scores', fontsize=16, fontweight='bold')
    # plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Methods', title_fontsize='13', fontsize='12', loc='upper right')

    # Adjust the layout
    plt.tight_layout()

    # Save the new histogram plot
    plt.savefig(f"{outdir}/blosum_histogram.pdf", dpi=300)
    print(f"Histogram plot saved at {outdir}/blosum_histogram.pdf")

    # Close the figure to free up memory
    plt.close()


def predict_random(data_csv, outdir, use_mhc=False):
    # Define a function to generate random peptide sequences
    def generate_random_peptide(length=9):
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # 20 standard amino acids
        return ''.join(random.choices(amino_acids, k=length))

    df = pd.read_csv(data_csv)
    if use_mhc:
        df[['tcr', 'MHC']] = df['text'].str.split('|', expand=True)
    else:
        try:
            df['tcr'] = df['text']
            df['epitope'] = df['label']
        except:
            pass

    # Create the 'epitope' column by applying the function
    df['pred_0'] = df.apply(lambda x: generate_random_peptide(), axis=1)

    Path(outdir).mkdir(parents=True, exist_ok=True)
    if use_mhc:
        df[['tcr', 'MHC', 'label', 'epitope']].to_csv(f"{outdir}/rand_pred.csv", index=False)
    else:
        df[['tcr', 'epitope', 'pred_0']].to_csv(f"{outdir}/rand_pred.csv", index=False)
    print(f"{outdir}/rand_pred.csv was saved. ")


def plot_length_distribution(pred_csv, outdir, k=10, show_k=False):
    # Read the CSV file
    df = pd.read_csv(pred_csv)
    # Replace all NaN values with empty string
    df = df.fillna("")

    # Calculate the length of the ground truth epitopes
    df['length_epitope'] = df['epitope'].apply(len)

    # Extract description from the file name
    desc = Path(pred_csv).stem

    # Determine the number of predictions to plot
    topk = min(len([x for x in df.columns if x.startswith('pred')]), k)

    # Create separate plots for each prediction
    for i in range(topk):
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Calculate lengths for current prediction
        df[f'length_{i}'] = df[f'pred_{i}'].apply(len)

        # Plot natural epitopes on the left
        sns.histplot(df['length_epitope'],
                    bins=range(8, 14),
                    kde=True,
                    color='#c1bed6',
                    ax=ax1,
                    label='Natural',
                    alpha=0.7)
        ax1.set_title('Natural Epitopes', fontsize=14)

        # Plot generated epitopes on the right
        sns.histplot(df[f'length_{i}'],
                    # bins=range(8, 14),
                    bins=range(min(df[f'length_{i}']), max(df[f'length_{i}']) + 2),
                    kde=False,
                    color='#7eb4db',
                    ax=ax2,
                    label='EpiGen',
                    alpha=0.7)
        ax2.set_title('Generated Epitopes', fontsize=14)

        # Set labels and ticks for both subplots
        for ax in [ax1, ax2]:
            ax.set_xlabel('Peptide Length', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.tick_params(labelsize=10)

        # Set overall title
        if show_k:
            plt.suptitle(f'Epitope Length Distribution (top {i+1})', fontsize=16)
        else:
            plt.suptitle(f'Epitope Length Distribution', fontsize=16)

        # Adjust layout
        plt.tight_layout()

        # Save each plot
        Path(outdir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{outdir}/length_distribution_{desc}_top{i+1}.pdf",
                   format='pdf',
                   bbox_inches='tight')
        plt.close()


# Function to read the GLIPH2 convergence groups
def read_gliph_convergence(filepath, group_size_th=3):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    groups = {}
    for line in lines:
        parts = line.strip().split('\t')
        group_size = int(parts[0])
        if group_size > group_size_th:
            group_key = parts[1]
            tcrs = parts[2:]
            groups[group_key] = tcrs[0].split(" ")
    return groups

def blosum62_score(pair):
    a, b = pair
    return blosum62.get((a, b), blosum62.get((b, a), -4))

def sequence_distance(seq_pair):
    seq1, seq2 = seq_pair
    return sum(blosum62_score((a, b)) for a, b in zip(seq1, seq2))

def pairwise_blosum62_distance(peptides, n_proc=1):
    global blosum62
    blosum62 = substitution_matrices.load('BLOSUM62')

    n_peptides = len(peptides)
    distances = np.zeros((n_peptides, n_peptides))

    if n_proc > 1:
        with multiprocessing.Pool(processes=n_proc) as pool:
            results = pool.map(sequence_distance, combinations(peptides, 2))

        idx = np.triu_indices(n_peptides, k=1)
        distances[idx] = results
        distances = distances + distances.T
    else:
        for i, j in combinations(range(n_peptides), 2):
            distance = sequence_distance((peptides[i], peptides[j]))
            distances[i, j] = distance
            distances[j, i] = distance

    return distances


def kmer_featurize(peptides, k=3):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k))
    X = vectorizer.fit_transform(peptides)
    return X.toarray()


def visualize_peptides_for_TCR_groups(gliph_convergence, data, outdir, col, feature='blosum', n_proc=1, legend=True):
    """
    gliph_convergence: str
        The result of running GLIPH2 algorithm. This contains TCR motif groups.
    data: str
        The file that contains TCR and peptide information.
        This can be a preprocessed dataset such as VDJdb (col=epitope)
        or can also be a prediction file by EpiGen (col=pred_0)
    outdir: str
        The output directory
    col: str
        The column to retrieve peptide information
    """
    assert col in ['label', 'epitope', 'pred_0']
    Path(outdir).mkdir(parents=True, exist_ok=True)
    desc = Path(data).stem

    df_gliph = read_gliph_convergence(gliph_convergence)
    df = pd.read_csv(data)
    peptides = df[col].unique()
    try:
        peptide_to_tcrs = df.groupby(col)['CDR3b'].apply(list).to_dict()
    except:
        try:
            peptide_to_tcrs = df.groupby(col)['tcr'].apply(list).to_dict()
        except:
            peptide_to_tcrs = df.groupby(col)['text'].apply(list).to_dict()

    if feature == 'blosum':
        dists = pairwise_blosum62_distance(peptides, n_proc)
    elif feature == 'kmer':
        kmer_features = kmer_featurize(peptides)
        dists = pairwise_distances(kmer_features, metric='euclidean')

    # Normalize the distances to [0, 1] range
    scaler = MinMaxScaler()
    dists_normalized = scaler.fit_transform(dists)
    umap = UMAP(metric='precomputed')  # Use precomputed metric since we're using distances
    peptide_2d = umap.fit_transform(dists_normalized)
    color = 'red'
    # Define the base size multiplier
    base_size = 10  # This will be multiplied by the number of TCRs

    # Create a legend scale that shows fewer representative values
    legend_scale = {2: 20, 10: 100, 20: 200, 30: 300, 50: 500}  # Representative values for legend

    # Function to get actual dot size (no upper limit)
    def get_dot_size(count):
        return count * base_size

    # Function to create size legend
    def add_size_legend(ax):
        legend_elements = [plt.scatter([], [], c=color, alpha=0.7, s=size,
                                     label=f'{count} TCRs')
                          for count, size in legend_scale.items()]
        ax.legend(handles=legend_elements, title='Number of TCRs',
                 title_fontsize='12', fontsize='10',
                 loc='upper right', bbox_to_anchor=(1.15, 1))

    # Calculate the number of rows and columns for the combined plot
    n_groups = len(df_gliph)
    n_cols = min(5, n_groups)  # Maximum 5 columns
    n_rows = math.ceil(n_groups / n_cols)

    # Create a figure for the combined plot
    fig_combined, axes_combined = plt.subplots(n_rows, n_cols, figsize=(10*n_cols, 10*n_rows))
    axes_combined = axes_combined.flatten() if n_groups > 1 else [axes_combined]

    for i, (group, tcrs) in enumerate(df_gliph.items()):
        # Create individual figure
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot background dots
        ax.scatter(peptide_2d[:, 0], peptide_2d[:, 1],
                  color='gray', s=20, alpha=0.5, label='All peptides')

        # Highlight cognate partner antigens
        group_peptides = [pep for pep in peptides if any(tcr in peptide_to_tcrs[pep] for tcr in tcrs)]
        indices = [np.where(peptides == pep)[0][0] for pep in group_peptides]

        # Calculate the number of TCRs for each peptide in this group
        tcr_counts = [sum(1 for tcr in tcrs if tcr in peptide_to_tcrs[pep]) for pep in group_peptides]

        # Use actual counts for sizes without upper limit
        sizes = [get_dot_size(count) for count in tcr_counts]

        # Plot highlighted dots
        scatter = ax.scatter(peptide_2d[indices, 0], peptide_2d[indices, 1],
                            color=color, s=sizes, alpha=0.7,
                            label=f'Group {group}')

        # Add size legend
        if legend:
            add_size_legend(ax)

        # Hide tick values
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(labelbottom=False, labelleft=False)

        # Adjust layout to accommodate legend
        plt.tight_layout()

        # Save the individual figure for this group
        filename = f'{outdir}/peps_{feature}_{col}_{desc}_group_{group}.pdf'
        plt.savefig(filename, dpi=300, bbox_inches='tight', format='pdf')
        plt.close(fig)

        # Now plot for the combined figure
        ax_combined = axes_combined[i]
        ax_combined.scatter(peptide_2d[:, 0], peptide_2d[:, 1],
                           color='gray', s=20, alpha=0.5, label='All peptides')
        ax_combined.scatter(peptide_2d[indices, 0], peptide_2d[indices, 1],
                           color=color, s=sizes, alpha=0.7,
                           label=f'Group {group}')

        # Add size legend to combined plot
        add_size_legend(ax_combined)

        # Hide tick values in combined plot
        ax_combined.set_xticks([])
        ax_combined.set_yticks([])
        ax_combined.tick_params(labelbottom=False, labelleft=False)

    # Remove any unused subplots in the combined figure
    for j in range(i+1, len(axes_combined)):
        fig_combined.delaxes(axes_combined[j])

    plt.tight_layout()

    # Save the combined figure
    combined_filename = f'{outdir}/peps_{feature}_{col}_{desc}_combined.pdf'
    plt.savefig(combined_filename, dpi=300, bbox_inches='tight', format='pdf')
    plt.close(fig_combined)

    print(f"All individual figures and the combined figure have been saved in {outdir}")


# Define a fixed order for amino acids
fixed_amino_acid_order = list('ACDEFGHIKLMNPQRSTVWY')


def draw_amino_acid_hist(pred_csv, outdir, topk=10, num_cols=2, show_k=False):
    # Read the CSV file
    df = pd.read_csv(pred_csv)

    # Extract description from the file name
    desc = Path(pred_csv).stem

    # Determine the number of top-k predictions to consider
    num_topk_preds = min(len([col for col in df.columns if col.startswith('pred_')]), topk)
    num_rows = math.ceil((num_topk_preds + 1) / num_cols)

    # Initialize the plot with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 10, num_rows * 6), constrained_layout=True)
    axes = axes.flatten()

    # Function to get amino acid counts
    def get_aa_counts(sequences):
        aa_counts = Counter(''.join(sequences.dropna().tolist()))
        aa_df = pd.DataFrame.from_dict(aa_counts, orient='index', columns=['Count']).reset_index()
        aa_df.columns = ['Amino Acid', 'Count']
        return aa_df.set_index('Amino Acid').reindex(fixed_amino_acid_order).reset_index().fillna(0)

    # Get natural epitope counts
    natural_aa_df = get_aa_counts(df['epitope'])

    # Plot amino acid usage for each prediction column
    for k in range(num_topk_preds):
        ax = axes[k]
        generated_aa_df = get_aa_counts(df[f'pred_{k}'])

        x = np.arange(len(fixed_amino_acid_order))

        # Calculate metrics
        # For JSD: Convert counts to probabilities
        natural_probs = natural_aa_df['Count'] / natural_aa_df['Count'].sum()
        generated_probs = generated_aa_df['Count'] / generated_aa_df['Count'].sum()

        # Calculate Jensen-Shannon Divergence
        jsd = jensenshannon(natural_probs, generated_probs)

        # Calculate Pearson Correlation
        correlation, p_value = pearsonr(natural_aa_df['Count'], generated_aa_df['Count'])

        # Save metrics to file
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, f'distribution_metrics_top{k+1}.txt'), 'w') as f:
            f.write(f"Jensen-Shannon Divergence: {jsd:.6f}\n")
            f.write(f"Pearson Correlation Coefficient: {correlation:.6f}\n")
            f.write(f"Correlation P-value: {p_value:.6f}\n")

        # Plot natural epitopes (slightly to the left)
        ax.bar(x - 0.2, natural_aa_df['Count'],
               width=0.4, color='purple', alpha=0.7, label='Natural')

        # Plot generated epitopes (slightly to the right)
        ax.bar(x + 0.2, generated_aa_df['Count'],
               width=0.4, color='blue', alpha=0.7, label='EpiGen')

        if show_k:
            ax.set_title(f'Amino Acid Usage (top {k+1})')
        else:
            ax.set_title(f'Amino Acid Usage')
        ax.set_xlabel('Amino Acid')
        ax.set_ylabel('Count')
        ax.legend()

        # Set x-axis ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels(fixed_amino_acid_order)

    # Remove any empty subplots
    for ax in axes[num_topk_preds:]:
        ax.remove()

    # Overall title
    # plt.suptitle('Amino Acid Usage in Natural and Generated Epitopes', fontsize=16)

    # Save the plot
    Path(outdir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{outdir}/amino_acid_usage_{desc}.pdf", format='pdf')
    plt.close()


def calculate_diversity_metrics(epitopes):
    epitope_counts = Counter(epitopes)
    N = len(epitopes)
    p = np.array(list(epitope_counts.values())) / N

    # Shannon Diversity Index
    Shannon = -np.sum(p * np.log(p))

    # Simpson's Diversity Index
    Simpson = 1 - np.sum(p ** 2)

    # Renyi Diversity Index (example for alpha = 2)
    alpha = 2
    Renyi = (1 / (1 - alpha)) * np.log(np.sum(p ** alpha))

    return Shannon, Simpson, Renyi

def rarefaction(epitopes, sample_size, iterations=1000):
    rarefied_metrics = []
    for _ in tqdm(range(iterations)):
        subsample = random.sample(epitopes, sample_size)
        metrics = calculate_diversity_metrics(subsample)
        rarefied_metrics.append(metrics)

    # Average the metrics across all iterations
    rarefied_metrics = np.array(rarefied_metrics)
    mean_metrics = np.mean(rarefied_metrics, axis=0)
    return mean_metrics


def measure_epitope_div(model_name, pred_csvs, datasets, outdir, sample_size=12000):
    # Created in 2024-10-25
    metrics_dict = {'Metric': [], 'Value': [], 'Dataset': []}
    for pred_csv, dataset in zip(pred_csvs, datasets):
        df = pd.read_csv(pred_csv)
        epitopes = df['pred_0'].tolist()

        # Basic diversity metrics
        pep2tcr_ratio = len(set(epitopes)) / len(epitopes)
        mean_metrics = rarefaction(epitopes, sample_size)
        metrics_dict['Metric'].extend(['Shannon', 'Simpson', 'Renyi', 'pep2tcr_ratio'])
        metrics_dict['Value'].extend(mean_metrics)
        metrics_dict['Value'].append(pep2tcr_ratio)
        metrics_dict['Dataset'].extend([dataset] * 4)

        # Additional metric: Average repetition of top 1% redundant peptides
        peptide_counts = Counter(epitopes)
        sorted_counts = sorted(peptide_counts.values(), reverse=True)
        top_1_percent_count = int(len(sorted_counts) * 0.01) or 1
        top_redundant_counts = sorted_counts[:top_1_percent_count]
        avg_repetition_top_1_percent = sum(top_redundant_counts) / top_1_percent_count
        metrics_dict['Metric'].append('avg_repetition_top_1_percent')
        metrics_dict['Value'].append(avg_repetition_top_1_percent)
        metrics_dict['Dataset'].append(dataset)

        # New metric: Top 10% concentration of total occurrences
        # Calculate the total number of occurrences
        total_occurrences = sum(peptide_counts.values())

        # Calculate the occurrences of the top 10% most frequent peptides
        top_10_percent_count = int(len(sorted_counts) * 0.10) or 1
        top_10_percent_occurrences = sum(sorted_counts[:top_10_percent_count])

        # Calculate the concentration ratio
        top_10_concentration = top_10_percent_occurrences / total_occurrences

        # Add this metric to metrics_dict
        metrics_dict['Metric'].append('top_10_concentration')
        metrics_dict['Value'].append(top_10_concentration)
        metrics_dict['Dataset'].append(dataset)

    # Create a DataFrame from the metrics and save it
    result = pd.DataFrame(metrics_dict)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    result.to_csv(f"{outdir}/diversity_{model_name}.csv", index=False)
    print(f"{outdir}/diversity_{model_name}.csv")
    return result


def plot_epitope_div(outdir, div_files, descs):
    # Created in 2024-10-25
    # Visualize Renyi, Simpson, Gini, Shannon, etc. for each model (method) in 3x2 subplots
    data = []
    for div_file, desc in zip(div_files, descs):
        df = pd.read_csv(div_file)
        df['Model'] = desc
        data.append(df)

    # Concatenate all dataframes
    df = pd.concat(data)

    # Pivot the data for plotting
    data_table = df.pivot_table(values='Value', index='Metric', columns='Model', aggfunc='mean')

    # Reorder columns of data_table to match the order in descs
    data_table = data_table[descs]

    # Set up 3x2 grid for subplots
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    axs = axs.flatten()  # Flatten for easy indexing
    colors = plt.get_cmap('viridis', len(descs)).colors

    # Iterate over each metric and plot in a separate subplot
    for i, metric in enumerate(data_table.index):
        ax = axs[i]
        bar_width = 0.25
        bar_positions = np.arange(len(descs))

        # Plot bars for each model
        for j, model in enumerate(data_table.columns):
            ax.bar(bar_positions[j] + (j * bar_width), data_table.loc[metric, model],
                   color=colors[j], width=bar_width, label=model if i == 0 else "")

        # Customize each subplot
        ax.set_title(metric, fontweight='bold')
        ax.set_xticks([])  # Remove x-ticks
        ax.set_xlabel('')   # Remove x-axis label

    # Adjust layout and add a legend
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    fig.legend(labels=data_table.columns, loc='upper right', bbox_to_anchor=(1.15, 0.9), title='Model')

    # Save the figure
    filename = 'diversity_subplots.pdf'
    plt.savefig(f"{outdir}/{filename}", dpi=300, format='pdf', bbox_inches='tight')
    print(f"{outdir}/{filename}")
    plt.close()


def plot_epitope_div_simple(outdir, div_files, descs):
    """
    Create a single plot comparing Renyi, Shannon, Simpson, and pep2tcr_ratio metrics
    across different models with side-by-side bars.
    """
    # Read and combine data
    data = []
    for div_file, desc in zip(div_files, descs):
        df = pd.read_csv(div_file)
        df['Model'] = desc
        data.append(df)
    df = pd.concat(data)

    # Filter only the metrics we want
    metrics_to_plot = ['Renyi', 'Shannon', 'Simpson', 'pep2tcr_ratio']
    df = df[df['Metric'].isin(metrics_to_plot)]

    # Create the plot
    plt.figure(figsize=(10, 10))

    # Setup bar positions
    n_models = len(descs)
    bar_width = 0.8 / n_models  # Divide available space by number of models
    metric_positions = np.arange(len(metrics_to_plot))

    # Plot bars for each model
    for i, model in enumerate(descs):
        model_data = df[df['Model'] == model]
        model_values = []
        for metric in metrics_to_plot:
            value = model_data[model_data['Metric'] == metric]['Value'].values
            model_values.append(value[0] if len(value) > 0 else 0)

        position = metric_positions + (i * bar_width) - (bar_width * (n_models-1)/2)
        plt.bar(position, model_values, bar_width, label=model)

    # Customize the plot
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Diversity Metrics Comparison')
    plt.xticks(metric_positions, metrics_to_plot)
    plt.legend(loc='upper right')

    # Save the plot
    plt.tight_layout()
    filename = 'diversity_comparison.pdf'
    plt.savefig(f"{outdir}/{filename}", dpi=300, format='pdf', bbox_inches='tight')
    print(f"{outdir}/{filename}")
    plt.close()


def measure_generation_redundancy(pred_csvs, descs, outdir):
    def _measure(df, idx):
        # Measure how many distinct epitopes there are in the top `idx` predictions
        values = []
        for row in df.iterrows():
            preds = row[1][[f'pred_{i}' for i in range(idx)]]
            values.append(len(set(preds)))
        return sum(values) / len(values)

    result = []
    for n in tqdm([1, 5, 10, 20]):
        for pred_csv, desc in zip(pred_csvs, descs):
            df = pd.read_csv(pred_csv)
            v = _measure(df, n)
            result.append((desc, v, n))

    df = pd.DataFrame(result, columns=['Model', 'Value', 'Number'])

    # Calculate the average 'Value' for each model and number
    df_avg = df.groupby(['Model', 'Number'])['Value'].mean().reset_index()

    # Plot the data
    plt.figure(figsize=(10, 6))
    for model in df_avg['Model'].unique():
        model_df = df_avg[df_avg['Model'] == model]
        plt.plot(model_df['Number'], model_df['Value'], label=model, marker='o')

    plt.xlabel('Number')
    plt.ylabel('Value')
    plt.title('TopK Generation Redundancy Per Model')
    plt.legend(title='Model')
    plt.grid(True)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{outdir}/pred_redundancy_comp.pdf", format='pdf')
    print(f"{outdir}/pred_redundancy_comp.pdf")
