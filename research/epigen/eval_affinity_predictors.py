# Standard library imports
import os
import pickle
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Local/application imports
from tabr_bert_fork.bert_pmhc import BERT as pmhc_net
from tabr_bert_fork.bert_tcr import BERT as tcr_net
from tabr_bert_fork.tcr_pmhc_model import (
    peptide_make_data,
    tcr_make_data,
    tcr_pmhc,
)


def evaluate_ensemble_critique(tcr_ckpt, pep_ckpt, tcr_pep_ckpts, test_data, outdir, desc, batch_size=512,
                               pep_d_model=256, tcr_d_model=256, tcr_maxlen=30, pep_maxlen=18, device='cuda'):
    """
    Evaluate the ensemble of TABR-BERT variants (TABR-BERT-mo)
    """
    n_gpus = torch.cuda.device_count()

    def load_model(model_class, checkpoint, device, maxlen, flag=None):
        model = model_class(pmhc_maxlen=maxlen) if flag == 'tcr_pmhc' else model_class(device=device, maxlen=maxlen)
        model = nn.DataParallel(model, list(range(n_gpus)))
        model.load_state_dict(torch.load(checkpoint))
        if device:
            model.to(device)
        model.eval()
        return model

    # Load the models
    tcr_model = load_model(tcr_net, tcr_ckpt, device, maxlen=tcr_maxlen)
    pep_model = load_model(pmhc_net, pep_ckpt, device, maxlen=pep_maxlen)
    models = [load_model(tcr_pmhc, ckpt, device, maxlen=pep_maxlen, flag='tcr_pmhc') for ckpt in tcr_pep_ckpts]

    # Load the dataset
    df_test = pd.read_csv(test_data)

    # Generate dataloaders
    tcr_loader = tcr_make_data(df_test['tcr'].tolist())
    pep_loader = peptide_make_data(df_test['epitope'].tolist())

    def extract_features(loader, model, output_file, feature_dim, maxlen):
        output = torch.Tensor()
        for input_ids in tqdm(loader, desc=f"Extracting features from {output_file}"):
            with torch.no_grad():
                input_ids = [i.cuda() for i in input_ids] if device == 'cuda' else input_ids
                model_output = model(*input_ids)
                reshaped_output = model_output[1] if isinstance(model_output, tuple) else model_output
                reshaped_output = reshaped_output.reshape(-1, maxlen * feature_dim).cpu()
                output = torch.cat([output, reshaped_output], dim=0)
        torch.save(output, output_file)
        print(f"Saved {output_file}")
        return output

    tcr_output = extract_features(tcr_loader, tcr_model, '__pycache__/tcr_output_test.pt', tcr_d_model, tcr_maxlen)
    pep_output = extract_features(pep_loader, pep_model, '__pycache__/peptide_output_test.pt', pep_d_model, pep_maxlen)

    # Predict the binding affinity
    inputs = torch.cat([tcr_output, pep_output], dim=-1)
    targets = torch.tensor(df_test['label'].values)
    data = TensorDataset(inputs, targets)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    softmax = nn.Softmax(dim=1)
    all_preds, all_labels = [], []

    for x, y in tqdm(loader, desc="Running inference using ensemble critique.."):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            preds = [softmax(model(x))[:, 1] for model in models]
        avg_pred = sum(preds) / len(preds)
        all_preds.extend(avg_pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    evaluate_and_plot(all_preds, all_labels, outdir, desc)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(f"{outdir}/ensemble_critique_{desc}.pkl", "wb") as f:
        result = {"all_preds": all_preds, "all_labels": all_labels}
        pickle.dump(result, f)
    return all_preds, all_labels


def evaluate_panpep(panpep, test_data, outdir, desc):
    """
    Evaluate PanPep model on test data.

    Parameters:
    panpep (str): Path to the PanPep script.
    test_data (str): Path to the test data CSV file.
    outdir (str): Directory to save the evaluation results.
    desc (str): Description for the evaluation plots.
    """
    original_cwd = os.getcwd()
    panpep_dir = os.path.dirname(panpep)

    # Prepare test data
    df = pd.read_csv(test_data).rename(columns={"tcr": "CDR3", "epitope": "Peptide"})
    all_labels = df['label'].tolist()

    cache_dir = Path(panpep_dir) / "__pycache__/PanPep"
    cache_file = cache_dir / "test_data.csv"
    pred_file = cache_dir / "pred.csv"

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    print(f"{cache_file} was saved.")

    # Run PanPep prediction
    try:
        os.chdir(panpep_dir)
        cmd = ['python', os.path.basename(panpep), '--learning_setting', 'zero-shot', '--input', str(cache_file), '--output', str(pred_file)]
        subprocess.run(cmd, check=True)
    finally:
        os.chdir(original_cwd)
        cache_file.unlink()

    # Read the output file and evaluate
    df = pd.read_csv(pred_file)
    all_preds = df['Score'].tolist()
    evaluate_and_plot(all_preds, all_labels, outdir, desc=desc)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(f"{outdir}/panpep_{desc}.pkl", "wb") as f:
        result = {"all_preds": all_preds, "all_labels": all_labels}
        pickle.dump(result, f)
    return all_preds, all_labels


def evaluate_teim(teim_seq, test_data, outdir, desc):
    # Save the current working directory
    original_cwd = os.getcwd()
    teim_dir = os.path.dirname(os.path.dirname(teim_seq))

    # Format test_data
    df = pd.read_csv(test_data)
    df = df.rename(columns={"tcr": "cdr3"})
    Path(f"{teim_dir}/__pycache__/testsets").mkdir(parents=True, exist_ok=True)
    cache_file = f"{teim_dir}/__pycache__/testsets/test.tsv"
    df.to_csv(cache_file, index=False, sep="\t")
    print(f"{cache_file} was saved. ")
    all_labels = df['label'].tolist()

    # Run TEIM-seq prediction
    try:
       # Change to the PanPep directory
       os.chdir(teim_dir)

       # Run PanPep prediction
       cmd = ['python', 'scripts/inference_seq.py']
       subprocess.run(cmd, check=True)

    finally:
        # Change back to the original working directory
        os.chdir(original_cwd)
        os.remove(cache_file)

    # Read the output file and evaluate
    df = pd.read_csv(f"{teim_dir}/outputs/sequence_level_binding.csv")
    all_preds = df['binding'].tolist()
    evaluate_and_plot(all_preds, all_labels, outdir, desc=desc)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(f"{outdir}/teim_{desc}.pkl", "wb") as f:
        result = {"all_preds": all_preds, "all_labels": all_labels}
        pickle.dump(result, f)
    return all_preds, all_labels


def evaluate_pmtnet(pmtnet, test_data, outdir, desc):
    # Save the current working directory
    original_cwd = os.getcwd()
    pmtnet_dir = os.path.dirname(pmtnet)

    # Format test_data
    df = pd.read_csv(test_data)
    df = df.rename(columns={"tcr": "CDR3", "epitope": "Antigen"})
    all_labels = df['label'].tolist()
    Path(f"{pmtnet_dir}/__pycache__").mkdir(parents=True, exist_ok=True)
    cache_file = f"{pmtnet_dir}/__pycache__/test.csv"
    df.to_csv(cache_file, index=False)
    print(f"{cache_file} was saved. ")

    # Run pMTnet prediction
    try:
       # Change to the PanPep directory
       os.chdir(pmtnet_dir)

       # Run PanPep prediction
       cmd = ['python', 'pMTnet.py', '-input', '__pycache__/test.csv', '-library', 'library', '-output', '__pycache__', '-output_log', '__pycache__/output.log']
       subprocess.run(cmd, check=True)
    except:
        return -1, -1, -1

    # Change back to the original working directory
    os.chdir(original_cwd)
    os.remove(cache_file)

    # Read the data
    df_pred = pd.read_csv(f'{pmtnet_dir}/__pycache__/prediction.csv')  # CDR3,Antigen,HLA,Rank
    df_test = pd.read_csv(test_data)  # tcr,epitope,label

    # Merge df_test to df_pred based on CDR3:tcr and Antigen:epitope with strict matching
    df_merged = df_pred.merge(df_test, left_on=['CDR3', 'Antigen', 'HLA'], right_on=['tcr', 'epitope', 'HLA'], how='inner')

    # Ensure only the necessary columns are kept and renamed correctly if needed
    df_merged = df_merged[['CDR3', 'Antigen', 'HLA', 'Rank', 'label']]

    # Convert labels and predictions to lists
    all_labels = df_merged['label'].tolist()
    df_merged['pred'] = 1 - df_merged['Rank']
    all_preds = df_merged['pred'].tolist()

    # Evaluate and plot the results
    auroc, auprc, f1 = evaluate_and_plot(all_preds, all_labels, outdir, desc=desc)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(f"{outdir}/pmtnet_{desc}.pkl", "wb") as f:
        result = {"all_preds": all_preds, "all_labels": all_labels}
        pickle.dump(result, f)
    return all_preds, all_labels


def evaluate_tabr_bert(tabr_bert, test_data, outdir, desc):
    # Save the current working directory
    original_cwd = os.getcwd()
    tabr_bert_dir = os.path.dirname(tabr_bert)

    # Format test_data
    df = pd.read_csv(test_data)
    df = df.rename(columns={"tcr": "cdr3", "epitope": "peptide", "HLA": "allele"})
    # Format the allele name
    df = df.dropna()
    df['allele'] = df['allele'].apply(lambda x: "HLA-" + str(x))
    all_labels = df['label'].tolist()
    Path(f"{tabr_bert_dir}/__pycache__").mkdir(parents=True, exist_ok=True)
    cache_file = f"{tabr_bert_dir}/__pycache__/test.csv"
    df.to_csv(cache_file, index=False)
    print(f"{cache_file} was saved. ")

    # Run TABR-BERT prediction
    try:
       # Change to the PanPep directory
       os.chdir(tabr_bert_dir)

       # Run PanPep prediction
       cmd = ['python', 'predict_tcr_pmhc_binding.py', '--input', '__pycache__/test.csv', '--GPUs', '1']
       subprocess.run(cmd, check=True)

    finally:
        # Change back to the original working directory
        os.chdir(original_cwd)
        os.remove(cache_file)

    # Read the data
    df_pred = pd.read_csv(f'{tabr_bert_dir}/output/output.csv')  # CDR3,Antigen,HLA,Rank

    # Convert labels and predictions to lists
    all_labels = df_pred['label'].tolist()
    all_preds = df_pred['rank'].tolist()

    # Evaluate and plot the results
    auroc, auprc, f1 = evaluate_and_plot(all_preds, all_labels, outdir, desc=desc)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(f"{outdir}/tabr_bert_{desc}.pkl", "wb") as f:
        result = {"all_preds": all_preds, "all_labels": all_labels}
        pickle.dump(result, f)
    return all_preds, all_labels


def evaluate_random(test_data, outdir, desc):
    import random
    # Load the test data
    df = pd.read_csv(test_data)

    # Extract labels
    all_labels = df['label'].tolist()

    # Generate random predictions
    all_preds = [random.random() for _ in range(len(df))]

    evaluate_and_plot(all_preds, all_labels, outdir, desc=desc)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(f"{outdir}/random_{desc}.pkl", "wb") as f:
        result = {"all_preds": all_preds, "all_labels": all_labels}
        pickle.dump(result, f)
    return all_preds, all_labels


def evaluate_per_HLA(eval_function: Callable, func_args: Dict[str, str], outdir: str):
    """
    eval_function: Callable
        The evaluation function to run.
    func_args: dict[str, str]
        The arguments for the eval_function.
    outdir: str
        The output directory to save the results.
    """
    # Read the test data
    df_test = pd.read_csv(func_args['test_data'])

    # Get counts of each HLA type
    hla_counts = df_test['HLA'].value_counts()

    results = []

    # Iterate over each HLA and run evaluation
    for hla, count in hla_counts.items():
        # Create a temporary test file
        cache_file = "__pycache__/per_HLA/test_hla.csv"
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)

        # Filter the test data for the current HLA and save to cache file
        df_test[df_test['HLA'] == hla].to_csv(cache_file, index=False)

        # Update the test_data argument
        func_args['test_data'] = cache_file

        # Run the evaluation function
        res = eval_function(**func_args)
        results.append((hla, count, res[0], res[1], res[2]))
        try:
            os.remove(cache_file)
        except:
            pass

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results, columns=['HLA', 'Count', 'AUROC', 'AUPRC', 'F1'])

    # Create output directory if it doesn't exist
    Path(outdir).mkdir(parents=True, exist_ok=True)
    # Save the results to a CSV file
    results_file = os.path.join(outdir, 'evaluation_results.csv')
    df_results.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")


def evaluate_and_plot(all_preds, all_labels, outdir, desc=''):
    """
    Evaluate and plot precision-recall and ROC curves, and calculate other metrics.

    Parameters:
    all_preds (array-like): Predicted probabilities.
    all_labels (array-like): True labels.
    outdir (str): Output directory for saving the plots.
    desc (str): Description for the plots.
    """
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Draw precision-recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    average_precision = average_precision_score(all_labels, all_preds)
    auprc = auc(recall, precision)  # Calculate AUPRC

    # Calculate and plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    # Calculate F1 score
    f1 = f1_score(all_labels, all_preds.round())

    # Calculate mean predicted probability and standard deviation
    mean_pred_prob = np.mean(all_preds)
    std_pred_prob = np.std(all_preds)

    # Print metrics
    print(f'Mean Predicted Probability for Positive Samples: {mean_pred_prob:.2f}')
    print(f'Standard Deviation of Predicted Probabilities: {std_pred_prob:.2f}')

    # Print metrics
    print(f'F1 Score: {f1:.2f}')
    print(f'AUROC: {roc_auc:.2f}')
    print(f'Average Precision (AP): {average_precision:.2f}')
    print(f'AUPRC: {auprc:.2f}')

    return roc_auc, auprc, f1
    #sns.set(style="whitegrid")
    #plt.figure(figsize=(10, 6))
    #sns.lineplot(x=recall, y=precision, marker='o', label=f'AP = {average_precision:.2f}')
    #plt.xlabel('Recall')
    #plt.ylabel('Precision')
    #plt.title(f'Precision-Recall Curve {desc}')
    #plt.legend(loc='lower left')
    #plt.savefig(f"{outdir}/precision-recall_{desc}.pdf", format='pdf')
    #plt.close()

    #plt.figure(figsize=(10, 6))
    #sns.lineplot(x=fpr, y=tpr, marker='o', label=f'AUC = {roc_auc:.2f}')
    #plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)  # y=x line
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title(f'ROC Curve {desc}')
    #plt.legend(loc='lower right')
    #plt.savefig(f"{outdir}/ROC_{desc}.pdf", format='pdf')
    #plt.close()


def evaluate_and_plot_all(pred_label_pkls, outdir, plot_only=False, df_path_pr=None, df_path_roc=None):
    # Evaluate the models on a dataset with a specific negative sample generation condition
    # ex) Evaluate on the VDJdb test set with negative shuffling
    MODEL_NAMES = {
        'ensemble_critique': 'EC',
        'panpep': 'PanPep',
        'pmtnet': 'pMTnet',
        'random': 'Random',
        'tabr_bert': 'TABR-BERT',
        'teim': 'TEIM'
    }

    def _parse(pkl):
        filename = str(Path(pkl).stem)
        for model_name in MODEL_NAMES.keys():
            if model_name in filename:
                dataset = filename.split(model_name + "_")[1]
                return MODEL_NAMES[model_name], dataset
        return -1, -1

    def save_curve_data(outdir, dataset, curve_type, data):
        rows = []
        for curve_data in data:
            x, y, metric, model_name, pkl = curve_data
            for i in range(len(x)):
                rows.append({
                    'model_name': model_name,
                    'x': x[i],
                    'y': y[i],
                    f'{curve_type}_metric': metric,
                    'pkl': pkl
                })

        df = pd.DataFrame(rows)
        filename = f"{outdir}/df_{curve_type}_{dataset}.csv"
        df.to_csv(filename, index=False)
        print(f"{filename} was saved.")

    def read_curve_data(file_path):
        df = pd.read_csv(file_path)
        curve_data = defaultdict(lambda: defaultdict(list))

        for _, row in df.iterrows():
            model_name = row['model_name']
            curve_data[model_name]['x'].append(row['x'])
            curve_data[model_name]['y'].append(row['y'])
            curve_data[model_name]['metric'] = row['pr_metric' if 'pr_metric' in df.columns else 'roc_metric']
            curve_data[model_name]['pkl'] = row['pkl']

        reconstructed_curves = []
        for model_name, data in curve_data.items():
            x = np.array(data['x'])
            y = np.array(data['y'])
            metric = data['metric']
            pkl = data['pkl']
            reconstructed_curves.append((x, y, metric, model_name, pkl))

        return reconstructed_curves

    Path(outdir).mkdir(parents=True, exist_ok=True)
    if plot_only:
        all_pr_curves = read_curve_data(df_path_pr)
        all_roc_curves = read_curve_data(df_path_roc)
    else:
        all_pr_curves = []
        all_roc_curves = []

        for pkl in tqdm(pred_label_pkls):
            with open(pkl, 'rb') as f:
                data = pickle.load(f)
            model_name, dataset = _parse(pkl)
            if dataset == 'Glanville' and model_name == 'pMTnet':
                # Skip this as pMTnet included Glanville in their training set
                continue
            preds = data['all_preds']
            labels = data['all_labels']

            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(labels, preds)
            average_precision = average_precision_score(labels, preds)
            auprc = auc(recall, precision)
            all_pr_curves.append((recall, precision, average_precision, model_name, pkl))

            # ROC Curve
            fpr, tpr, _ = roc_curve(labels, preds)
            roc_auc = auc(fpr, tpr)
            all_roc_curves.append((fpr, tpr, roc_auc, model_name, pkl))

        # Save Precision-Recall curve data
        save_curve_data(outdir, dataset, 'pr', all_pr_curves)

        # Save ROC curve data
        save_curve_data(outdir, dataset, 'roc', all_roc_curves)

    print("Start drawing AUROC and AUPRC curves..")
    sp = pred_label_pkls[0].split("/")[-1]  # random_Glanville_test_ext_peptide_paired_neg_multi_1.pkl
    ind = sp.index("_")
    dataset = sp[:-4][ind+1:]
    dataset_name = sp.split("_")[1]
    if dataset_name == 'test':
        dataset_name = 'Combined test set'
    elif dataset_name == 'vdjdb':
        dataset_name = 'VDJdb'
    multi = int(sp[:-4].split("_")[-1])

    # Rename dataset_type
    if "test_ext_peptide_paired_neg_multi" in sp:
        dataset_type = "Neg pairs by external peptides"
    elif "test_with_neg_multi" in sp:
        dataset_type = "Neg pairs by shuffling"
    elif "test_ext_paired_neg_multi" in sp:
        dataset_type = "Neg pairs by external tcrs"

    if dataset_name == 'Glanville':
        for i, entries in enumerate(all_pr_curves):
            if entries[3] == 'pMTnet':
                del all_pr_curves[i]
        for j, entries in enumerate(all_roc_curves):
            if entries[3] == 'pMTnet':
                del all_roc_curves[i]
        print("Removed pMTnet result when evaluating on Glanville")

    # PRC plot
    plt.figure(figsize=(12, 8))
    for recall, precision, avg_precision, model_name, _ in all_pr_curves:
        sns.lineplot(x=recall, y=precision, label=f'{model_name} (AP = {avg_precision:.2f})')
    plt.xlabel('Rec', fontsize=16)
    plt.ylabel('Pre', fontsize=16)
    plt.title(f'PRC, {dataset_name}, {dataset_type}', fontsize=16)
    plt.legend(loc='lower left', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{outdir}/AUPRC_{dataset}.pdf", format='pdf')
    plt.close()

    # ROC plot
    plt.figure(figsize=(12, 8))
    for fpr, tpr, roc_auc, model_name, _ in all_roc_curves:
        sns.lineplot(x=fpr, y=tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=0.8)  # y=x line
    plt.xlabel('FPR', fontsize=16)
    plt.ylabel('TPR', fontsize=16)
    plt.title(f'ROC, {dataset_name}, {dataset_type}', fontsize=16)
    plt.legend(loc='lower right', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{outdir}/AUROC_{dataset}.pdf", format='pdf')
    plt.close()
