import os
import io
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pdf2image import convert_from_path
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors

import plotly.graph_objects as go


def merged_perc_rank(rank_pkls, methods, outdir, title):
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Load data from each pickle file
    data = []
    for pkl in rank_pkls:
        with open(pkl, "rb") as f:
            rank = pickle.load(f)
        data.append(rank)

    # Create a figure for the violin plots
    plt.figure(figsize=(12, 8))

    # Define colors for the violins
    colors = ['#FF6347', '#4682B4', '#32CD32']  # Tomato, SteelBlue, LimeGreen

    # Create violin plots for each dataset
    vplots = plt.violinplot(data, showmeans=False, showextrema=True, showmedians=True)

    # Customize the appearance of violin plots
    for i, body in enumerate(vplots['bodies']):
        body.set_facecolor(colors[i % len(colors)])  # Set different colors
        body.set_edgecolor('black')
        body.set_alpha(0.7)

    # Customize the median lines
    vplots['cmedians'].set_color('black')
    vplots['cmedians'].set_linewidth(2)

    # Customize the extrema lines
    vplots['cbars'].set_color('black')
    vplots['cbars'].set_linewidth(1.5)

    # Add labels, title, and grid
    plt.ylabel('Percentile Rank', fontsize=14)
    plt.title(f'Percentile Rank Distribution on {title}', fontsize=16)
    plt.xticks(range(1, len(methods) + 1), methods)  # Set x-ticks based on method names
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.gca().tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.ylim(0, 100)  # Adjust as needed based on data range

    # Save the plot to a file
    plt.savefig(f'{outdir}/perc_ranks_merged.pdf', dpi=300, format='pdf')
    plt.close()


def load_and_filter_dataframe(df_path, topk=1, n_controls=100):
    """
    Read and filter a DataFrame from a CSV file.

    Parameters:
    df_path (str): Path to the CSV file.
    topk (int): Value for TopK filter. Default is 1.
    n_controls (int): Value for NControls filter. Default is 100.

    Returns:
    pd.DataFrame: Filtered DataFrame.
    """
    df_raw = pd.read_csv(df_path)
    return df_raw[(df_raw['TopK'] == topk) & (df_raw['NControls'] == n_controls)]


def plot_percentile_rank_bars(dataframes, descs, outdir, figsize=(12, 6), dpi=300, filename='combined_perc_rank.pdf', orient='v', mode='T1234'):
    """
    For T1234 and public test set plots with consistent bar widths.
    """
    sns.set_palette("deep")
    plt.figure(figsize=figsize)

    # Common width for bars in both modes
    bar_width = 0.3  # You can adjust this value

    # Combine all dataframes into one for seaborn
    combined_df = pd.concat([df.assign(Dataset=desc) for df, desc in zip(dataframes, descs)], ignore_index=True)

    if mode == 'T1234':
        # Plot for T1234 mode with specified width
        ax = sns.barplot(
            x='Dataset',
            y='PercentileRank',
            data=combined_df,
            capsize=0.4,
            width=bar_width,  # Use common width
            dodge=False,
            orient=orient
        )
        ax.set_ylim(60, 100)
        plt.xlabel('Pseudo-labeled test set', fontsize=16)
        for i in ax.containers:
            ax.bar_label(i, fmt='%.1f', padding=16)
    else:
        # Parse function remains the same
        def parse(x):
            if 'EpiGen' in x:
                return 'EpiGen'
            elif x == 'random_pred':
                return 'RandGen'
            elif x == 'knn_pred':
                return 'KNN'
            elif 'processed' in x:
                return 'EpiGen'
            else:
                return x

        combined_df['Prediction'] = combined_df['Prediction'].apply(lambda x: parse(x))

        # Calculate width for grouped bars
        n_groups = len(combined_df['Prediction'].unique())
        group_width = bar_width * n_groups
        ax = sns.barplot(
            x='Dataset',
            y='PercentileRank',
            hue='Prediction',
            data=combined_df,
            capsize=0.1,
            width=group_width * 0.5,  # Use calculated group width
            dodge=True,
            orient=orient
        )
        ax.set_ylim(0, 100)
        plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
        plt.xlabel('Test set', fontsize=16)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=16)

    # Rest of the customization remains the same
    plt.ylabel('Mean Percentile Rank of Binding Affinity', fontsize=16)
    plt.title('Percentile Rank Comparison', fontsize=14)
    plt.xticks(rotation=0, ha='center', fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(outdir, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"Plot saved as {filepath}")


def plot_multiple_dataframes(filename, df_paths, descs, outdir, topk=1, n_controls=100, figsize=(10, 10), dpi=300, orient='v', mode='T1234'):
    """
    Wrapper function to plot multiple datasets from CSV files in the same plot.

    Parameters:
    df_paths (list): List of paths to CSV files.
    descs (list): List of descriptions for each dataset.
    outdir (str): Output directory for the plot.
    topk (int): Value for TopK filter. Default is 1.
    n_controls (int): Value for NControls filter. Default is 100.
    figsize (tuple): Figure size.
    dpi (int): DPI for the output image.

    Returns:
    None
    """
    if len(df_paths) != len(descs):
        raise ValueError("The number of file paths must match the number of descriptions.")

    # Load and filter all DataFrames
    dataframes = [load_and_filter_dataframe(path, topk, n_controls) for path in df_paths]

    # Plot the data
    plot_percentile_rank_bars(dataframes, descs, outdir, figsize, dpi, orient=orient, filename=filename, mode=mode)


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
                f"processed_{dataset}_test": "EpiGen"
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
                f"EpiGen_{dataset}": "EpiGen"
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


def edf5_comparative(indir="figures_e24_re/3f/EpiGen", outdir="figures_e24_re/3f/EpiGen/compare"):
    """
    Create plots that compares the peptide clustering plots
    by Natural and EpiGen, handling PDF input files
    """
    def parse_filename(x):
        if "label" in x:
            model = "label"
        else:
            model = "EpiGen"
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
        if 'label' in info and 'EpiGen' in info:
            dataset, filename_label = info['label']
            _, filename_EpiGen = info['EpiGen']

            # Create a new figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

            # Load and plot the 'label' PDF on the left
            pages_label = convert_from_path(os.path.join(indir, filename_label))
            img_label = pages_label[0]  # Assume single page PDF
            ax1.imshow(img_label)
            ax1.axis('off')
            ax1.set_title(f'Dataset: {dataset}', fontsize=25, loc='center', pad=20)

            # Load and plot the 'EpiGen' PDF on the right
            pages_EpiGen = convert_from_path(os.path.join(indir, filename_EpiGen))
            img_EpiGen = pages_EpiGen[0]  # Assume single page PDF
            ax2.imshow(img_EpiGen)
            ax2.axis('off')
            ax2.set_title('EpiGen-generated', fontsize=25, loc='center', pad=20)

            # Set the main title for the entire figure
            fig.suptitle(f"Epitopes paired with TCRs motif group: {tcr}", fontsize=25)

            # Adjust the layout and save the figure
            plt.tight_layout()
            output_filename = f"comparison_{dataset}_{tcr}.pdf"
            plt.savefig(os.path.join(outdir, output_filename))
            plt.close(fig)

    print(f"Comparative figures have been saved in {outdir}")


def fig1b_array(plots, outdir):
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
