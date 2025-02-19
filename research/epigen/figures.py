# Standard library imports
import io
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union
from collections import Counter

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
import plotly.graph_objects as go
from tqdm import tqdm
from scipy import stats
import statsmodels.stats.multitest

# Image processing imports
from PIL import Image
from pdf2image import convert_from_path

# Specialized visualization
import logomaker


def fig1d_sankey_diagram(category2num_before, category2num_after, outdir):
    # List of categories in a fixed order
    category_order = ["Virus", "Bacteria", "Self", "Tumor", "Parasite", "Fungi", "Archaea", "Others"]

    # Define colors for each category
    category_colors = {
        "Virus": "rgba(255, 127, 14, 0.6)",     # Orange
        "Bacteria": "rgba(31, 119, 180, 0.6)",  # Blue
        "Self": "rgba(44, 160, 44, 0.6)",       # Green
        "Tumor": "rgba(214, 39, 40, 0.6)",      # Red
        "Parasite": "rgba(148, 103, 189, 0.6)", # Purple
        "Fungi": "rgba(140, 86, 75, 0.6)",      # Brown
        "Archaea": "rgba(227, 119, 194, 0.6)",  # Pink
        "Others": "rgba(127, 127, 127, 0.6)"    # Gray
    }

    # Calculate the total for normalization
    total_before = sum(category2num_before.values())
    total_after = sum(category2num_after.values())

    # Normalize proportions
    proportions_before = {cat: count / total_before for cat, count in category2num_before.items()}
    proportions_after = {cat: count / total_after for cat, count in category2num_after.items()}

    # Create node labels with proportions included
    labels = [
        f"{cat} ({proportions_before.get(cat, 0) * 100:.1f}%)" for cat in category_order
    ] + ["Antigen Category Filter"] + [
        f"{cat} ({proportions_after.get(cat, 0) * 100:.1f}%)" for cat in category_order
    ]

    # Link sources and targets
    source_before_to_intermediate = list(range(len(category_order)))  # from "before" nodes
    target_before_to_intermediate = [len(category_order)] * len(category_order)  # to single intermediate node

    source_intermediate_to_after = [len(category_order)] * len(category_order)  # from single intermediate node
    target_intermediate_to_after = [i + len(category_order) + 1 for i in range(len(category_order))]  # to "after" nodes

    # Define link values
    link_values_before = [proportions_before.get(cat, 0) for cat in category_order]
    link_values_after = [proportions_after.get(cat, 0) for cat in category_order]

    # Define link colors based on categories
    link_colors = [
        category_colors[cat] for cat in category_order
    ] * 2  # duplicate for both before and after links

    # Combine all links for the Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=[category_colors[cat] for cat in category_order] + ["rgba(169, 169, 169, 0.6)"] + [category_colors[cat] for cat in category_order]
        ),
        link=dict(
            source=source_before_to_intermediate + source_intermediate_to_after,
            target=target_before_to_intermediate + target_intermediate_to_after,
            value=link_values_before + link_values_after,
            color=link_colors,
        )
    ))

    # Update layout and save as PDF
    fig.update_layout(title_text="Antigen Category Filter: Before and After", font_size=10)
    fig.write_image(f"{outdir}/sankey_diagram.pdf", format="pdf")


def fig2cdf(outdir, df_summary, metrics_pkl, parameters_pkl):
    """
    Draw the figures using df_summary
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    dataset = str(Path(outdir).stem)
    df = pd.read_csv(df_summary)
    with open(metrics_pkl, "rb") as f:
        metrics = pickle.load(f)
    with open(parameters_pkl, "rb") as f:
        parameters = pickle.load(f)

    def draw_line_plots(outdir, dataset, metrics, pred_csv_names, n_control_values, topk_values):
        # Summarize the results and plot
        summary_data = []
        for topk in topk_values:
            for pred_csv in pred_csv_names:
                mets = metrics[topk][pred_csv]
                for n in n_control_values:
                    percentile_ranks = [m[2] for sublist in mets for m in sublist if m[0] == n]
                    for pr in percentile_ranks:
                        summary_data.append((topk, pred_csv, n, pr))
        df_summary = pd.DataFrame(summary_data, columns=['TopK', 'Prediction', 'NControls', 'PercentileRank'])
        df_summary.to_csv(f"{outdir}/df_summary.csv", index=False)
        print(f'{outdir}/df_summary.csv')

        # Plot: Percentile Rank vs TopK (Fixed Number of Controls)
        fixed_n = n_control_values[-1]
        fixed_data = []
        for topk in topk_values:
            for pred_csv in pred_csv_names:
                mets = metrics[topk][pred_csv]
                percentile_ranks = [m[2] for sublist in mets for m in sublist if m[0] == fixed_n]
                for pr in percentile_ranks:
                    fixed_data.append((topk, pred_csv, pr))
        df_fixed = pd.DataFrame(fixed_data, columns=['TopK', 'Prediction', 'PercentileRank'])

        # Calculate mean percentile rank for each TopK and Prediction combination
        df_mean = df_fixed.groupby(['TopK', 'Prediction'])['PercentileRank'].agg(['mean', 'std']).reset_index()

        # Set up the plot
        plt.figure(figsize=(12, 10))

        # Define colors to match the image
        colors = {
            'random_pred': '#eda6b3',  # red
            'knn_pred': '#89bd79',  # green
            # f'EpiGen_{dataset}': '#85bbe2'  # blue
            f"processed_{dataset}_test": '#85bbe2'  # blue
        }

        # Define custom legend labels
        legend_labels = {
            "knn_pred": "KNN",
            "random_pred": "RandGen",
            # f"EpiGen_{dataset}": "EpiGen"
            f"processed_{dataset}_test": "EpiGen"
        }

        # Plot each Prediction type
        for pred in df_mean['Prediction'].unique():
            data = df_mean[df_mean['Prediction'] == pred]
            plt.plot(data['TopK'], data['mean'], marker='o', linestyle='-',
                     linewidth=2, markersize=5, label=legend_labels[pred], color=colors[pred])

            # Add narrower error bars
            plt.fill_between(data['TopK'], data['mean'] - 0.1 * data['std'],
                             data['mean'] + 0.1 * data['std'], alpha=0.2, color=colors[pred])

        # Customize the plot
        plt.title(f'{dataset}', fontsize=16)
        plt.xlabel('Top K-th generation', fontsize=16)
        plt.ylabel('Mean Percentile Rank of Affinity', fontsize=16)

        # Set custom x-axis ticks
        plt.xticks([1, 5, 10, 20], ['1', '5', '10', '20'], fontsize=14)

        # Increase tick label font size (y-axis)
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)

        # Set y-axis range (adjust as needed based on your data)
        plt.ylim(45, 100)

        # Customize the legend
        plt.legend(fontsize='16', bbox_to_anchor=(0.98, 0.98), loc='upper right', borderaxespad=0.)

        # Remove grid
        # plt.grid(True, linestyle='--', alpha=0.7)  # This line is now commented out

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'{outdir}/lineplot_topk_mean.pdf', dpi=300, bbox_inches='tight', format='pdf')
        plt.close()

    def draw_hist_of_affinity(outdir, dataset, metrics, pred_csv_names, n_control_values, topk_values):
        fixed_n = n_control_values[-1]
        for topk in topk_values:
            # Prepare data
            affinity_data = []
            rank_data = []
            for pred_csv in pred_csv_names:
                mets = metrics[topk][pred_csv]
                for sublist in mets:
                    for m in sublist:
                        if m[0] == fixed_n:
                            affinity_data.append((pred_csv, m[3], 'Generated Affinity'))
                            affinity_data.append((pred_csv, m[4], 'Random Affinity'))
                            rank_data.append((pred_csv, m[2]))

            df_affinity = pd.DataFrame(affinity_data, columns=['Prediction', 'Affinity', 'Type'])
            df_rank = pd.DataFrame(rank_data, columns=['Prediction', 'Percentile Rank'])

            colors = {
                'random_pred': '#eda6b3',  # red
                'knn_pred': '#89bd79',  # green
                # f'EpiGen_{dataset}': '#85bbe2'  # blue
                f"processed_{dataset}_test": '#85bbe2'
            }

            legend_labels = {
                "knn_pred": "KNN",
                "random_pred": "RandGen",
                # f"EpiGen_{dataset}": "EpiGen"
                f"processed_{dataset}_test": "EpitopeGen"
            }

            # Create a figure with 1x3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            fig.suptitle(f'{dataset}', fontsize=16)

            # Initialize max_count to store the maximum histogram count
            max_count = 0

            # First loop: Find the maximum count for setting y-axis limits
            for pred_csv in pred_csv_names:
                # Get data for this prediction
                pred_data = df_affinity[(df_affinity['Type'] == 'Generated Affinity') &
                                        (df_affinity['Prediction'] == pred_csv)]

                # Calculate histogram values without plotting
                counts, _ = np.histogram(pred_data['Affinity'], bins=30)
                max_count = max(max_count, counts.max())

            # Plot each prediction in a separate subplot
            for i, pred_csv in enumerate(pred_csv_names):
                # Get data for this prediction
                pred_data = df_affinity[(df_affinity['Type'] == 'Generated Affinity') &
                                      (df_affinity['Prediction'] == pred_csv)]

                # Plot histogram in the corresponding subplot
                sns.histplot(data=pred_data,
                            x='Affinity',
                            ax=axes[i],
                            label=legend_labels.get(pred_csv, pred_csv),
                            kde=False,
                            color=colors[pred_csv],
                            bins=30)

                # Customize each subplot
                axes[i].set_xlabel('Affinity', fontsize=16)
                axes[i].set_ylabel('Frequency', fontsize=16)
                axes[i].set_ylim(0, int(max_count*1.1))
                axes[i].set_title(legend_labels.get(pred_csv, pred_csv), fontsize=16)
                axes[i].tick_params(labelsize=16)

            plt.tight_layout()
            plt.savefig(f'{outdir}/affinity_distribution_values_topk_{topk}.pdf',
                       dpi=300,
                       format='pdf',
                       bbox_inches='tight')
            plt.close()

    def draw_violin_plots(outdir, dataset, metrics, pred_csv_names, n_control_values, topk_values):
        fixed_n = n_control_values[-1]
        for topk in topk_values:
            affinity_data = []
            rank_data = []
            for pred_csv in pred_csv_names:
                mets = metrics[topk][pred_csv]
                for sublist in mets:
                    for m in sublist:
                        if m[0] == fixed_n:
                            affinity_data.append((pred_csv, m[3], 'Generated Affinity'))
                            affinity_data.append((pred_csv, m[4], 'Random Affinity'))
                            rank_data.append((pred_csv, m[2]))  # Append percentile rank

            df_affinity = pd.DataFrame(affinity_data, columns=['Prediction', 'Affinity', 'Type'])
            df_rank = pd.DataFrame(rank_data, columns=['Prediction', 'Percentile Rank'])

            legend_labels = {
                "knn_pred": "KNN",
                "random_pred": "RandGen",
                f"EpiGen_{dataset}": "EpitopeGen"
            }

            # Define the color palette outside the loop (if not already defined)
            palette = sns.color_palette("husl", n_colors=len(pred_csv_names))

            # Violin plot for each topk - Percentile Rank
            plt.figure(figsize=(10, 10))

            # Create a mapping dictionary for the DataFrame
            label_map = {old: new for old, new in legend_labels.items() if old in df_rank['Prediction'].unique()}

            # Apply the mapping to the DataFrame
            df_rank['Prediction'] = df_rank['Prediction'].map(label_map).fillna(df_rank['Prediction'])

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

            plt.title(f'{dataset}', fontsize=16)
            # Remove x-axis label
            plt.xlabel('')

            # Add labels and title
            plt.ylabel('Percentile Rank of Affinity', fontsize=18)
            plt.gca().tick_params(axis='both', which='major', labelsize=18)

            plt.tight_layout()
            plt.savefig(f'{outdir}/percentile_rank_distribution_topk_{topk}.pdf', dpi=300, format='pdf')
            plt.close()

    pred_csv_names = parameters["pred_csv_names"]
    n_control_values = parameters["n_control_values"]
    topk_values = parameters["topk_values"]

    draw_line_plots(outdir, dataset, metrics, pred_csv_names, n_control_values, topk_values)
    draw_violin_plots(outdir, dataset, metrics, pred_csv_names, n_control_values, topk_values)
    draw_hist_of_affinity(outdir, dataset, metrics, pred_csv_names, n_control_values, topk_values)
    return


def edf5_comparative(indir="figures_e24_re/3f/EpitopeGen", outdir="figures_e24_re/3f/EpitopeGen/compare"):
    """
    Create plots that compares the peptide clustering plots
    by Natural and EpitopeGen, handling PDF input files
    """
    def parse_filename(x):
        if "label" in x:
            model = "label"
        else:
            model = "EpitopeGen"
        if "IEDB" in x:
            dataset = "IEDB"
        elif "VDJdb" in x:
            dataset = "VDJdb"
        elif "PIRD" in x:
            dataset = "PIRD"
        elif "McPAS" in x:
            dataset = "McPAS-TCR"
        tcr = x[:-4].split("-")[1]
        return model, dataset, tcr

    data = {}
    for filename in os.listdir(indir):
        if not os.path.isfile(os.path.join(indir, filename)):
            continue
        if "combined" in filename:
            continue
        model, dataset, tcr = parse_filename(filename)
        if tcr in data:
            data[tcr][model] = (dataset, filename)
        else:
            data[tcr] = {model: (dataset, filename)}

    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    for tcr, info in tqdm(data.items()):
        if 'label' in info and 'EpitopeGen' in info:
            dataset, filename_label = info['label']
            _, filename_EpitopeGen = info['EpitopeGen']

            # Create a new figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

            # Load and plot the 'label' PDF on the left
            pages_label = convert_from_path(os.path.join(indir, filename_label))
            img_label = pages_label[0]  # Assume single page PDF
            ax1.imshow(img_label)
            ax1.axis('off')
            ax1.set_title(f'Dataset: {dataset}', fontsize=25, loc='center', pad=20)

            # Load and plot the 'EpitopeGen' PDF on the right
            pages_EpitopeGen = convert_from_path(os.path.join(indir, filename_EpitopeGen))
            img_EpitopeGen = pages_EpitopeGen[0]  # Assume single page PDF
            ax2.imshow(img_EpitopeGen)
            ax2.axis('off')
            ax2.set_title('EpitopeGen-generated', fontsize=25, loc='center', pad=20)

            # Set the main title for the entire figure
            fig.suptitle(f"Epitopes paired with TCRs motif group: {tcr}", fontsize=25)

            # Adjust the layout and save the figure
            plt.tight_layout()
            output_filename = f"comparison_{dataset}_{tcr}.pdf"
            plt.savefig(os.path.join(outdir, output_filename))
            plt.close(fig)

    print(f"Comparative figures have been saved in {outdir}")


def fig1b_array(plots, outdir):
    """
    AUROC, AUPRC array plots
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    n_rows = len(plots)  # Number of rows (6 in this case)
    n_cols = len(plots[0])  # Number of columns (3 in this case)

    # Load all images and measure dimensions to calculate figure size dynamically
    images = []
    max_width = 0
    total_height = 0

    for row in plots:
        row_images = []
        row_height = 0
        row_width = 0
        for filepath in row:
            img = convert_from_path(filepath)[0]  # Load first page of PDF
            row_images.append(img)
            width, height = img.size
            row_width += width
            row_height = max(row_height, height)  # Take the maximum height in the row
        max_width = max(max_width, row_width)  # Update max_width with the widest row
        total_height += row_height  # Accumulate total height
        images.append(row_images)

    # Set the overall figure size based on total dimensions
    dpi = 100  # Resolution
    fig_width = max_width / dpi
    fig_height = total_height / dpi

    # Adjust spacing between subplots using gridspec_kw
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_width, fig_height), dpi=dpi,
        gridspec_kw={'wspace': 0.01, 'hspace': 0.01}  # Adjust spaces between subplots
    )

    # Display each image in the corresponding subplot
    for row_idx, row_images in enumerate(images):
        for col_idx, img in enumerate(row_images):
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.axis('off')  # Turn off axis for a cleaner look

    # Save the integrated figure as a single PDF
    output_filepath = f"{outdir}/aggregated_plot.pdf"
    plt.savefig(output_filepath, format='pdf', bbox_inches='tight')

    # Close the figure to free up memory
    plt.close(fig)


def plot_species_distribution(outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    # Data
    species = ['Virus', 'Bacteria', 'Self', 'Tumor', 'Parasite', 'Archaea']
    # percentages = [69.7, 13.7, 12.6, 2.7, 0.4, 0.9]  # EpiGen_e28
    # percentages = [73.1, 14.2, 7.7, 2.9, 1.3, 0.8]
    percentages = [75.8, 13.8, 7.4, 1.7, 0.9, 0.4]

    # Colors for each category
    colors = ['#FF7F0E', '#1F77B4', '#2CA02C', '#D62728', '#9467BD', '#E377C2']

    # Create pie chart
    plt.figure(figsize=(10, 8))
    patches, texts, autotexts = plt.pie(percentages,
                                      labels=species,
                                      colors=colors,
                                      autopct='%1.1f%%',
                                      startangle=90)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')

    # Add title
    plt.title('Distribution of Antigen Categories', pad=20)

    # Add legend with sample counts
    legend_labels = [f'{s} ({c:,})' for s, c in zip(species, percentages)]
    plt.legend(patches, legend_labels,
              title="Antigen Category Distribution",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()

    # Show plot
    plt.savefig(f"{outdir}/piechart.pdf", format='pdf')



### 2025-02-13 Improving design
### Fig. 3b. Amino acid usages
def calculate_aa_frequencies(sequences):
    """Calculate amino acid frequencies from a list of sequences."""
    aa_counts = Counter(''.join(sequences))
    total = sum(aa_counts.values())
    return {aa: (count/total)*100 for aa, count in aa_counts.items()}

def get_aa_properties():
    """Return amino acid properties for grouping."""
    return {
        'hydrophobic': ['A', 'I', 'L', 'M', 'F', 'W', 'V'],
        'polar': ['N', 'Q', 'S', 'T', 'Y'],
        'positive': ['H', 'K', 'R'],
        'negative': ['D', 'E'],
        'special': ['C', 'G', 'P']
    }

def create_aa_composition_viz(df1, df2, df3, output_file="aa_composition.pdf",
                            labels=["Natural", "Model 1", "Model 2"]):
    """
    Create publication-quality visualizations comparing amino acid compositions of three datasets,
    with amino acids grouped by their properties.
    """
    # Calculate frequencies
    nat_freqs = calculate_aa_frequencies(df1['epitope'])
    model1_freqs = calculate_aa_frequencies(df2['pred_0'])
    model2_freqs = calculate_aa_frequencies(df3['pred_0'])

    # Get AA properties
    aa_props = get_aa_properties()

    # Create ordered list of AAs based on properties
    ordered_aas = []
    group_boundaries = []
    group_centers = []
    current_position = 0

    # Order for property groups
    property_order = ['hydrophobic', 'polar', 'positive', 'negative', 'special']

    for prop in property_order:
        aas = sorted(aa_props[prop])
        ordered_aas.extend(aas)
        current_position += len(aas)
        group_boundaries.append(current_position)
        group_centers.append(current_position - len(aas)/2)

    # Color palette for three datasets
    colors = ['#c1bed6', '#85bbe2', '#e3b93b']

    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    # Upper panel: Enhanced bar plot
    ax1 = fig.add_subplot(gs[0])
    x = np.arange(len(ordered_aas))
    width = 0.25  # Narrower bars for three groups

    # Create bars
    model1_heights = [model1_freqs.get(aa, 0) for aa in ordered_aas]
    model2_heights = [model2_freqs.get(aa, 0) for aa in ordered_aas]
    nat_heights = [nat_freqs.get(aa, 0) for aa in ordered_aas]

    rects1 = ax1.bar(x - width, nat_heights, width, label=labels[0],
                     color=colors[0], alpha=0.7)
    rects2 = ax1.bar(x, model1_heights, width, label=labels[1],
                     color=colors[1], alpha=0.7)
    rects3 = ax1.bar(x + width, model2_heights, width, label=labels[2],
                     color=colors[2], alpha=0.7)

    # Customize upper panel
    ax1.set_ylabel('Frequency (%)', fontsize=16, fontweight='bold')
    ax1.set_title('A) Amino Acid Usage Comparison', loc='left',
                  fontsize=16, fontweight='bold')

    # Set x-ticks and labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(ordered_aas, fontweight='bold')
    # Increase tick label font size (y-axis)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    # Add vertical lines between groups
    prev_boundary = -0.5
    for i, boundary in enumerate(group_boundaries[:-1]):
        ax1.axvline(x=boundary - 0.5, color='gray', linestyle='--', alpha=0.3)

        # Add property group labels
        center = (prev_boundary + boundary - 0.5) / 2
        ax1.text(center, ax1.get_ylim()[1] * 1.05, property_order[i].capitalize(),
                horizontalalignment='center', fontsize=16, fontweight='bold')
        prev_boundary = boundary

    # Add last group label
    center = (prev_boundary + len(ordered_aas) - 0.5) / 2
    ax1.text(center, ax1.get_ylim()[1] * 1.05, property_order[-1].capitalize(),
             horizontalalignment='center', fontsize=16, fontweight='bold')

    # Customize legend and grid
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, axis='y', alpha=0.3)

    # Add light background colors for different property groups
    colors_bg = ['#f8f9fa', '#f0f3f5']  # Alternating very light colors
    prev_boundary = -0.5
    for i, boundary in enumerate(group_boundaries):
        ax1.axvspan(prev_boundary, boundary - 0.5,
                    color=colors_bg[i % 2], alpha=0.3, zorder=0)
        prev_boundary = boundary - 0.5

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


### Fig. 2a. T1234 plot
def create_multi_boxplot(csv_files, labels=None, metric_col='PercentileRank',
                        output_file='multi_boxplot.pdf',
                        title='Binding Affinity Comparison',
                        figsize=(8, 6)):  # Reduced figure width
    """
    Create a sophisticated publication-quality box plot visualization for multiple CSV files.

    Parameters:
    -----------
    csv_files : list
        List of paths to CSV files
    labels : list, optional
        Labels for each dataset. If None, will use filenames
    metric_col : str, default='PercentileRank'
        Name of the column containing the metric to plot
    output_file : str, default='multi_boxplot.pdf'
        Path to save the output visualization
    title : str, default='Distribution Comparison'
        Title for the plot
    figsize : tuple, default=(8, 6)
        Figure size in inches
    """
    # Set style
    plt.style.use('seaborn-whitegrid')

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    all_data = []
    all_labels = []
    all_values = []

    if labels is None:
        labels = [Path(f).stem for f in csv_files]

    # Custom color palette - more sophisticated colors
    colors = sns.color_palette("Blues", n_colors=len(csv_files))

    # Read and combine data
    for i, (csv_file, label) in enumerate(zip(csv_files, labels)):
        df = pd.read_csv(csv_file)
        values = df[metric_col].values
        all_values.append(values)
        all_data.extend(values)
        all_labels.extend([label] * len(values))

    # Create DataFrame for seaborn
    plot_df = pd.DataFrame({
        'Value': all_data,
        'Group': all_labels
    })

    # Create enhanced box plot
    bp = sns.boxplot(data=plot_df, x='Group', y='Value', ax=ax,
                    palette=colors, width=0.6,
                    boxprops={'alpha': 0.8, 'linewidth': 2},
                    showfliers=True,  # Show outliers
                    flierprops={'marker': 'o', 'markerfacecolor': 'gray',
                              'markersize': 4, 'alpha': 0.5},
                    medianprops={'color': 'white', 'linewidth': 2},
                    whiskerprops={'linewidth': 2},
                    capprops={'linewidth': 2})

    # Add means with enhanced styling
    # for i, (label, values) in enumerate(zip(labels, all_values)):
    #     mean = np.mean(values)

    #     # Add mean value text
    #     ax.text(i, mean + 1, f'{mean:.1f}',
    #             ha='center', va='bottom', fontsize=16,
    #             bbox=dict(facecolor='white', edgecolor='none',
    #                      alpha=0.7, pad=2))

    # Enhance grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.grid(False)

    # Remove top and right spines
    sns.despine(ax=ax)

    # Increase font sizes for tick labels
    ax.tick_params(axis='both', which='major', labelsize=16)  # Increased tick label size

    # Add subtle background shading for alternating groups
    # for i in range(0, len(labels), 2):
    #     ax.axvspan(i-0.5, i+0.5, color='gray', alpha=0.1)

    # Customize axes
    ax.set_ylabel('Percentile Rank', fontsize=16, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=16, fontweight='bold')  # Remove x-label as it's redundant

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0, ha='center', fontweight='bold')

    # Add title
    plt.title(title, fontsize=16, fontweight='bold', pad=20)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


### Fig. 2b. Public datasets, external datasets
def create_model_boxplot_comparison(dataframes, dataset_names, outdir,
                                  filename='model_boxplot_comparison.pdf',
                                  figsize=(12, 7)):
    """
    Create a sophisticated box plot visualization comparing model performances across datasets.

    Parameters:
    -----------
    dataframes : list
        List of pandas DataFrames containing the data
    dataset_names : list
        List of dataset names (VDJdb, IEDB, etc.)
    outdir : str
        Output directory path
    filename : str
        Output filename
    figsize : tuple
        Figure size in inches
    """
    # Set style
    plt.style.use('seaborn-whitegrid')

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Combine dataframes
    combined_df = pd.concat([df.assign(Dataset=name)
                           for df, name in zip(dataframes, dataset_names)],
                          ignore_index=True)

    # Parse model names
    def parse_model_name(x):
        if 'EpiGen' in x or 'processed' in x:
            return 'EpitopeGen'
        elif x == 'random_pred':
            return 'RandGen'
        elif x == 'knn_pred':
            return 'BLOSUMGen'
        return x

    combined_df['Prediction'] = combined_df['Prediction'].apply(parse_model_name)

    # Set color palette
    colors = ['#eda6b3', '#89bd79', '#85bbe2']  # RandGen, KNN, EpitopeGen
    sns.set_palette(colors)

    # Create sophisticated box plot
    bp = sns.boxplot(data=combined_df,
                    x='Dataset',
                    y='PercentileRank',
                    hue='Prediction',
                    width=0.8,
                    boxprops={'alpha': 0.8, 'linewidth': 2},
                    showfliers=True,
                    flierprops={'marker': 'o', 'markerfacecolor': 'gray',
                               'markersize': 4, 'alpha': 0.5},
                    medianprops={'color': 'white', 'linewidth': 2},
                    whiskerprops={'linewidth': 2},
                    capprops={'linewidth': 2})

    # Calculate and add mean values
    # for dataset_idx, dataset in enumerate(dataset_names):
    #     dataset_data = combined_df[combined_df['Dataset'] == dataset]
    #     for model_idx, model in enumerate(sorted(dataset_data['Prediction'].unique())):
    #         model_data = dataset_data[dataset_data['Prediction'] == model]
    #         mean_val = model_data['PercentileRank'].mean()

    #         # Calculate position for mean annotation
    #         pos = dataset_idx + (model_idx - 1) * 0.3

    #         # Add mean value text
    #         ax.text(pos, mean_val + 1, f'{mean_val:.1f}',
    #                ha='center', va='bottom', fontsize=16,
    #                bbox=dict(facecolor='white', edgecolor='none',
    #                         alpha=0.7, pad=1))

    # Enhance grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.grid(False)

    # Remove top and right spines
    sns.despine()

    # Add subtle background shading for alternating datasets
    # for i in range(0, len(dataset_names), 2):
    #     ax.axvspan(i-0.5, i+0.5, color='gray', alpha=0.1)

    # Customize axes
    ax.set_ylabel('Percentile Rank', fontsize=16, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=16, fontweight='bold')

    # Customize legend
    legend = ax.legend(title='Model',
                      bbox_to_anchor=(1.02, 1),
                      loc='upper left',
                      fontsize=16)
    legend.get_title().set_fontweight('bold')

    # Set title
    plt.title('Binding Affinity Comparison',
             fontsize=16, fontweight='bold', pad=20)

    # Rotate x-axis labels for better readability if needed
    plt.xticks(rotation=0, fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16)

    # Adjust layout and save
    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(outdir, filename)

    # Save with high resolution
    plt.savefig(filepath, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()

    print(f"Plot saved as {filepath}")


### Fig. 2e, 2f. Diversity Radial Plot
def create_diversity_radar_plot(csv_files, labels=None, output_file='diversity_radar.pdf'):
    """
    Create a hexagonal radar plot for diversity metrics comparison.

    Parameters:
    -----------
    csv_files : list
        List of paths to CSV files containing diversity metrics
    labels : list, optional
        Labels for each dataset. If None, will use filenames
    output_file : str
        Path to save the output visualization
    """
    # Set style
    plt.style.use('seaborn-whitegrid')

    # Read and process all data first to get proper normalization
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Inverse the metrics where lower is better
    mask_invert = combined_df['Metric'].isin(['avg_repetition_top_1_percent', 'top_10_concentration'])
    combined_df.loc[mask_invert, 'Value'] = 1 / combined_df.loc[mask_invert, 'Value']
    for df in all_data:
        mask_invert = df['Metric'].isin(['avg_repetition_top_1_percent', 'top_10_concentration'])
        df.loc[mask_invert, 'Value'] = 1 / df.loc[mask_invert, 'Value']

    # Get max values for normalization
    max_values = {}
    for metric in combined_df['Metric'].unique():
        if metric in ['Simpson', 'pep2tcr_ratio']:
            max_values[metric] = 1.0  # These are already between 0 and 1
        else:
            max_values[metric] = combined_df[combined_df['Metric'] == metric]['Value'].max()
    max_values['Renyi'] = 304.16

    # Normalize values
    for idx in combined_df.index:
        metric = combined_df.loc[idx, 'Metric']
        combined_df.loc[idx, 'Value'] = combined_df.loc[idx, 'Value'] / max_values[metric]

    # Prepare for plotting
    metrics = ['Shannon', 'Simpson', 'Renyi', 'pep2tcr_ratio',
              'avg_repetition_top_1_percent', 'top_10_concentration']
    num_metrics = len(metrics)

    # Compute angle for each metric
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Close the polygon

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Color palette
    # colors = sns.color_palette("husl", n_colors=len(csv_files))
    # colors = ["#85BBE2", "#1E6AE6"]
    colors = ["#85BBE2", "#d66342"]
    # colors = sns.color_palette("Blues", n_colors=len(csv_files))
    # colors = ['#2196F3', '#1565C0', '#0D47A1', '#000000']
    # colors.reverse()

    # Plot data for each dataset
    for idx, df in enumerate(all_data):
        values = []
        for metric in metrics:
            value = df[df['Metric'] == metric]['Value'].values[0]
            value = value / max_values[metric]
            values.append(value)
        values = np.concatenate((values, [values[0]]))  # Close the polygon

        label = labels[idx] if labels else f'Dataset {idx+1}'
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])

    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=16, fontweight='bold')

    # Add gridlines
    ax.set_rlim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    legend = ax.legend(title='Datasets', bbox_to_anchor=(1.2, 1.0),
                      fontsize=16, title_fontsize=16)
    legend.get_title().set_fontweight('bold')

    # Add title
    plt.title('Diversity Metrics Comparison', fontsize=16, fontweight='bold', pad=20)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(output_file)
    plt.close()


### Antigen Category Distribution, Biological Sanity Score
def biological_sanity_comparison_plot(models: Dict[str, Dict[str, float]],
                               output_path: str = 'model_comparison.pdf',
                               figsize: tuple = (12, 8)) -> None:
    """
    Create a sophisticated visualization comparing model distributions with BSS scores.

    Args:
        models: Dictionary of model names and their distributions
        output_path: Path to save the output figure
        figsize: Figure size as (width, height)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    def calculate_bss(distribution):
        reference = {
            'Virus': 69.7, 'Bacteria': 13.7, 'Self': 12.6, 'Tumor': 2.7,
            'Parasite': 0.4, 'Fungi': 0.0, 'Archaea': 0.9, 'Others': 0.0
        }
        weights = {
            'Virus': 5.0, 'Tumor': 3.0, 'Bacteria': 2.0, 'Self': 2.0,
            'Parasite': 1.0, 'Fungi': 0.5, 'Archaea': 0.2, 'Others': 0.3
        }
        weighted_diff_sum = sum(
            weights[category] * abs(distribution.get(category, 0) - ref_value)
            for category, ref_value in reference.items()
        )
        return np.exp(-weighted_diff_sum/100)

    # Define sophisticated color palette
    category_colors = {
        "Virus": "#FF7F0E",
        "Bacteria": "#1F77B4",
        "Self": "#2CA02C",
        "Tumor": "#D62728",
        "Parasite": "#9467BD",
        "Fungi": "#BA9A93",
        "Archaea": "#EEADDA",
        "Others": "#B2B2B2"
    }

    # Colors for each category
    colors = ['#FF7F0E', '#1F77B4', '#2CA02C', '#D62728', '#9467BD', '#E377C2']


    # Calculate BSS scores
    bss_scores = {model: calculate_bss(dist) for model, dist in models.items()}

    # Set style for publication quality
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 10,
        'axes.linewidth': 1.5,
        'axes.edgecolor': '#333333'
    })

    # Create figure with gridspec for custom layout
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, len(models), height_ratios=[1, 3], hspace=0.3)

    # Plot BSS scores on top row
    ax_bss = fig.add_subplot(gs[0, :])
    x = np.arange(len(models))
    scores = [bss_scores[model] for model in models.keys()]

    # Create sophisticated BSS score bars
    score_bars = ax_bss.bar(x, scores, color='#2ecc71', alpha=0.8, width=0.6,
                            edgecolor='white', linewidth=1.5)

    # Customize BSS plot
    ax_bss.set_ylim(0, 1)
    ax_bss.set_ylabel('Biological Coherence\nScore', fontsize=12, fontweight='bold')
    ax_bss.set_xticks([])
    ax_bss.spines['top'].set_visible(False)
    ax_bss.spines['right'].set_visible(False)
    ax_bss.grid(True, axis='y', alpha=0.3, linestyle='--')

    # Add score values with enhanced styling
    for i, score in enumerate(scores):
        ax_bss.text(i, score + 0.02, f'{score:.3f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    # Create pie charts for each model
    for idx, (model_name, distribution) in enumerate(models.items()):
        ax_pie = fig.add_subplot(gs[1, idx])

        # Prepare data for pie chart
        values = []
        colors = []
        labels = []
        explode = []

        for category, value in distribution.items():
            if value > 0:  # Only include non-zero values
                values.append(value)
                colors.append(category_colors[category])
                labels.append(f'{category}\n({value:.1f}%)' if value >= 5 else '')
                explode.append(0.05 if category == 'Virus' else 0)

        # Create sophisticated pie chart
        wedges, texts, autotexts = ax_pie.pie(values, explode=explode, colors=colors,
                                             labels=labels, autopct='',
                                             pctdistance=0.85, wedgeprops={'width': 0.7,
                                                                          'edgecolor': 'white',
                                                                          'linewidth': 1.5})

        # Add model name as title
        ax_pie.set_title(model_name, pad=20, fontsize=12, fontweight='bold')

    # Create common legend
    legend_elements = [Rectangle((0, 0), 1, 1, facecolor=color, label=category)
                      for category, color in category_colors.items()]
    fig.legend(handles=legend_elements, loc='center left',
              bbox_to_anchor=(1.02, 0.5), title='Categories',
              title_fontsize=12, fontsize=10)

    # Add title
    plt.suptitle('Model Comparison: Antigen Category Distribution',
                fontsize=14, fontweight='bold', y=0.95)

    # Save figure with high resolution
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_bss(distribution: Union[Dict[str, float], List[float]],
                  normalize: bool = True) -> float:
    """
    Calculate the Biological Category Alignment Score (BSS) for a given distribution.

    Args:
        distribution: Either a dictionary with category names as keys and proportions as values,
                     or a list of proportions in the order [Virus, Bacteria, Self, Tumor, Parasite,
                     Fungi, Archaea, Others]
        normalize: If True, normalize the input distribution to sum to 100

    Returns:
        float: BSS score between 0 and 1
    """
    # Reference distribution (in percentages)
    reference = {
        'Virus': 45.0,
        'Bacteria': 12.5,
        'Self': 10.0,
        'Tumor': 17.5,
        'Parasite': 6.5,
        'Fungi': 3.0,
        'Archaea': 0.5,
        'Others': 5.0
    }

    # Category weights
    weights = {
        'Virus': 5.0,
        'Tumor': 3.0,
        'Bacteria': 2.0,
        'Self': 2.0,
        'Parasite': 1.0,
        'Fungi': 0.5,
        'Archaea': 0.2,
        'Others': 0.3
    }

    # Convert list to dictionary if necessary
    if isinstance(distribution, list):
        categories = list(reference.keys())
        distribution = dict(zip(categories, distribution))

    # Normalize distribution if requested
    if normalize:
        total = sum(distribution.values())
        distribution = {k: (v/total * 100) for k, v in distribution.items()}

    # Calculate weighted sum of absolute differences
    weighted_diff_sum = sum(
        weights[category] * abs(distribution.get(category, 0) - ref_value)
        for category, ref_value in reference.items()
    )

    # Calculate BSS score
    bss = np.exp(-weighted_diff_sum/100)  # Divide by 100 to handle percentages

    return bss


### Logomaker plot
def normalize_tcr_length(sequence, target_length=15):
    """
    Normalize TCR sequence to target length by handling the conserved regions
    and sliding the middle variable region.

    Args:
        sequence (str): Input TCR sequence
        target_length (int): Desired length for normalization (default: 15)

    Returns:
        list: List of normalized subsequences that contribute to the position weight matrix
    """
    if len(sequence) < 7:  # Too short to be a valid TCR
        return []

    # Define conserved region lengths
    conserved_start = 4  # First 4 positions
    conserved_end = 4    # Last 4 positions

    # Handle sequences shorter than target length
    if len(sequence) <= target_length:
        # Extract middle variable region
        middle_start = conserved_start
        middle_end = len(sequence) - conserved_end
        middle_region = sequence[middle_start:middle_end]

        # Calculate padding needed
        total_padding = target_length - len(sequence)
        pad_left = total_padding // 2
        pad_right = total_padding - pad_left

        # Create normalized sequence with padding
        normalized = (
            sequence[:conserved_start] +
            '-' * pad_left +
            middle_region +
            '-' * pad_right +
            sequence[-conserved_end:]
        )
        return [normalized]

    # Handle sequences longer than target length
    else:
        normalized_sequences = []
        middle_start = conserved_start
        middle_end = len(sequence) - conserved_end
        middle_region = sequence[middle_start:middle_end]

        # Calculate sliding window size for middle region
        window_size = target_length - (conserved_start + conserved_end)

        # Slide through middle region
        for i in range(len(middle_region) - window_size + 1):
            window = middle_region[i:i+window_size]
            normalized = sequence[:conserved_start] + window + sequence[-conserved_end:]
            normalized_sequences.append(normalized)

        return normalized_sequences

def create_position_weight_matrix(sequences, target_length=15):
    """
    Create a position weight matrix from a list of TCR sequences.

    Args:
        sequences (list): List of TCR sequences
        target_length (int): Target length for normalization

    Returns:
        tuple: (position weight matrix as DataFrame, number of valid sequences)
    """
    if not sequences:
        return None, 0

    # Initialize counts
    counts = defaultdict(lambda: defaultdict(float))
    total_contributions = np.zeros(target_length)
    valid_seq_count = 0

    # Process each sequence
    for seq in sequences:
        normalized_seqs = normalize_tcr_length(seq, target_length)
        if normalized_seqs:
            valid_seq_count += 1
            weight = 1.0 / len(normalized_seqs)  # Distribute weight among multiple contributions

            # Count amino acids at each position
            for norm_seq in normalized_seqs:
                for pos, aa in enumerate(norm_seq):
                    if aa != '-':  # Skip padding characters
                        counts[pos][aa] += weight
                        total_contributions[pos] += weight

    if valid_seq_count == 0:
        return None, 0

    # Create position weight matrix
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    pwm_data = []

    for pos in range(target_length):
        pos_freqs = {}
        total = total_contributions[pos] if total_contributions[pos] > 0 else 1

        for aa in amino_acids:
            pos_freqs[aa] = counts[pos][aa] / total

        pwm_data.append(pos_freqs)

    # Convert to DataFrame
    pwm_df = pd.DataFrame(pwm_data)
    return pwm_df, valid_seq_count


def create_sequence_logo(ax, pwm_df, title):
    """
    Create a sequence logo plot using logomaker.

    Args:
        ax (matplotlib.axes.Axes): Axes object for plotting
        pwm_df (pd.DataFrame): Position weight matrix as DataFrame
        title (str): Title for the plot
    """
    if pwm_df is None:
        ax.text(0.5, 0.5, 'No valid sequences',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=16)
        ax.set_title(title, fontsize=16)
        return

    # Create logo
    logo = logomaker.Logo(pwm_df,
                         ax=ax,
                         fade_below=0.1,
                         shade_below=0.2,
                         width=0.9)

    # Style the logo
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    logo.style_xticks(rotation=0, fmt='%d', anchor=0)

    # Increase font sizes
    ax.tick_params(axis='both', which='major', labelsize=16)
    logo.style_xticks(fontsize=16)
    # logo.style_yticks(fontsize=16)

    # Customize axes with larger fonts
    ax.set_ylabel('Probability', fontsize=16)
    ax.set_title(title, fontsize=16, pad=20)

    # Add grid
    ax.grid(True, axis='y', alpha=0.2)

    # Set y-axis limits
    ax.set_ylim(0, 1)

def logo_comparison(prediction, vdjdb, outdir, top_k=8):
    """
    Create and save comparative sequence logo plots for predicted and reference TCRs.

    Args:
        prediction (str): Path to CSV file containing predictions
        vdjdb (str): Path to CSV file containing VDJdb reference data
        outdir (str): Output directory for saving plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    # Read input files
    df_pred = pd.read_csv(prediction)
    for i in range(top_k):
        df_pred[f'pred_{i}'] = df_pred[f'pred_{i}'].apply(lambda x: x[:9])
    print("Trim lengths down to 9..")
    df_vdjdb = pd.read_csv(vdjdb)

    # Most frequent epitopes in VDJdb database
    epitopes = df_vdjdb['epitope'].value_counts().head(10).index.tolist()

    for epitope in epitopes:
        # Get TCR sequences
        tcrs_pred = df_pred[df_pred[[f'pred_{i}' for i in range(top_k)]].eq(epitope).any(axis=1)]['tcr'].tolist()
        tcrs_ref = df_vdjdb[df_vdjdb['epitope'] == epitope]['tcr'].tolist()

        # Skip if no sequences found
        if not tcrs_pred and not tcrs_ref:
            print(f"No sequences found for epitope: {epitope}")
            continue

        # Create position weight matrices with length info
        pwm_pred, n_pred = create_position_weight_matrix(tcrs_pred)
        pwm_ref, n_ref = create_position_weight_matrix(tcrs_ref)

        if pwm_pred is None and pwm_ref is None:
            print(f"No valid sequences for epitope: {epitope}")
            continue

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Create logos
        create_sequence_logo(ax1, pwm_pred,
                           f'Predicted TCRs for {epitope}\n(n={n_pred})')
        create_sequence_logo(ax2, pwm_ref,
                           f'Reference TCRs for {epitope}\n(n={n_ref})')

        # Add statistics about length distributions
        plt.tight_layout()
        plt.savefig(f"{outdir}/logo_comparison_{epitope}.pdf",
                    dpi=300, bbox_inches='tight')
        plt.close()


### Fig. 3a
def plot_length_distribution(pred_csv, outdir, k=10, show_k=False, show_long_tail=False, color='#85bbe2'):
    """
    Plot length distribution of epitopes with optional long tail visualization.

    Args:
        pred_csv (str): Path to prediction CSV file
        outdir (str): Output directory for plots
        k (int): Number of top predictions to plot
        show_k (bool): Whether to show k in the title
        show_long_tail (bool): Whether to show long tail visualization in inset
    """
    # Read and prepare data
    df = pd.read_csv(pred_csv)
    df = df.fillna("")
    df['length_epitope'] = df['epitope'].apply(len)
    desc = Path(pred_csv).stem
    topk = min(len([x for x in df.columns if x.startswith('pred')]), k)

    for i in range(topk):
        # Calculate lengths for current prediction
        df[f'length_{i}'] = df[f'pred_{i}'].apply(len)

        if show_long_tail:
            # Create figure with main plot and prepare for inset
            fig = plt.figure(figsize=(15, 6))
            gs = fig.add_gridspec(1, 2)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax2_inset = ax2.inset_axes([0.5, 0.5, 0.45, 0.45])
        else:
            # Create simple figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot natural epitopes
        sns.histplot(df['length_epitope'],
                    bins=range(8, 14),
                    kde=False,
                    color='#c1bed6',
                    ax=ax1,
                    label='Natural',
                    alpha=0.7)
        ax1.set_title('Natural Epitopes', fontsize=14)

        # Plot generated epitopes (main plot)
        if show_long_tail:
            # Main plot shows normal range
            sns.histplot(df[f'length_{i}'],
                        bins=range(8, 14),
                        kde=False,
                        color='#c1bed6',
                        ax=ax2,
                        label='EpiGen',
                        alpha=0.7)

            # Plot long tail in inset (length > 12)
            long_tail_data = df[df[f'length_{i}'] > 12][f'length_{i}']
            if len(long_tail_data) > 0:
                sns.histplot(long_tail_data,
                            bins=range(13, int(max(long_tail_data)) + 2),
                            kde=False,
                            color=color,
                            ax=ax2_inset,
                            alpha=0.7)
                ax2_inset.set_title('Long Sequences (>12)', fontsize=10)
                # ax2_inset.grid(False, alpha=0.3)
                ax2_inset.set_yscale('log')
                ax2_inset.set_xlabel('Length', fontsize=10)
                ax2_inset.set_ylabel('Freq (log)', fontsize=10)
                ax2_inset.tick_params(labelsize=10)
        else:
            # Simple histogram for all lengths
            sns.histplot(df[f'length_{i}'],
                        bins=range(min(df[f'length_{i}']), max(df[f'length_{i}']) + 2),
                        kde=False,
                        color=color,
                        ax=ax2,
                        label='EpiGen',
                        alpha=0.7)

        ax2.set_title('Generated Epitopes', fontsize=14)

        # Style main plots
        for ax in [ax1, ax2]:
            ax.set_xlabel('Peptide Length', fontsize=20)
            ax.set_ylabel('Frequency', fontsize=20)
            ax.tick_params(labelsize=20)
            # ax.grid(False, alpha=0.3)

        # Set overall title
        if show_k:
            plt.suptitle(f'Epitope Length Distribution (top {i+1})', fontsize=16)
        else:
            plt.suptitle(f'Epitope Length Distribution', fontsize=16)

        # Adjust layout
        plt.tight_layout()

        # Save plot
        Path(outdir).mkdir(parents=True, exist_ok=True)
        suffix = '_with_tail' if show_long_tail else ''
        output_path = f"{outdir}/length_distribution_{desc}_top{i+1}{suffix}.pdf"
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Saved: {output_path}")
        plt.close()


### Fig. 3c
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
