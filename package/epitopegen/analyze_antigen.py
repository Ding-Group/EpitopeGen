import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod

class ProteinNameStandardizer(ABC):
    """Abstract base class for protein name standardization."""

    @abstractmethod
    def standardize(self, protein_name: str) -> str:
        """Standardize a protein name."""
        pass

class CoronavirusProteinStandardizer(ProteinNameStandardizer):
    """Standardizer for coronavirus protein names."""

    def standardize(self, protein_name: str) -> str:
        """Standardize coronavirus protein names.

        Args:
            protein_name: Raw protein name string.

        Returns:
            str: Standardized protein name.
        """
        if not isinstance(protein_name, str):
            return protein_name

        # Remove coronavirus strain information
        if '[Severe acute respiratory syndrome coronavirus 2]' in protein_name:
            protein_name = protein_name.split("[")[0].strip()

        # Standardize name
        protein_name = protein_name.lower()

        # Map to standard names
        if any(x in protein_name for x in ['orf1ab', 'replicase']):
            return 'Non-structural proteins (NSP)'
        elif any(x in protein_name for x in ['orf']):
            return 'Accessory proteins (ORFs)'
        elif any(x in protein_name for x in ['surface', 'spike', 'glycoprotein']):
            return 'Spike (S) protein'
        elif any(x in protein_name for x in ['nucleocapsid', 'nucleoprotein']):
            return 'Nucleocapsid (N) protein'
        elif any(x in protein_name for x in ['membrane']):
            return 'Membrane (M) protein'
        elif any(x in protein_name for x in ['envelope', 'envelop']):
            return 'Envelope (E) protein'
        else:
            return 'Other'


class AntigenAnalyzer:
    def __init__(
        self,
        condition_patterns: Dict[str, str],
        pattern_order: List[str],
        protein_colors: Dict[str, str],
        protein_standardizer: Optional[ProteinNameStandardizer] = None,
        cell_type_column: str = 'leiden',
        cell_type_mapping: Optional[Dict[str, str]] = None,
        output_dir: str = "analysis/antigen_distribution"
    ):
        """Initialize antigen distribution analyzer.

        Args:
            condition_patterns: Dictionary mapping condition values to pattern names.
            pattern_order: List defining the order of patterns for visualization.
            protein_colors: Dictionary mapping protein names to their colors.
            protein_standardizer: Object that implements protein name standardization.
                Defaults to CoronavirusProteinStandardizer if None.
            cell_type_column: Column name containing cell type information.
            cell_type_mapping: Optional dictionary mapping cell type ids to names.
            output_dir: Directory path to save analysis results.
        """
        self.condition_patterns = condition_patterns
        self.pattern_order = pattern_order
        self.protein_colors = protein_colors
        self.protein_standardizer = protein_standardizer or CoronavirusProteinStandardizer()
        self.cell_type_column = cell_type_column
        self.cell_type_mapping = cell_type_mapping or {}
        self.output_dir = Path(output_dir)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate inputs
        self._validate_inputs()

    def _get_matching_proteins(
        self,
        df: pd.DataFrame,
        top_k: int
    ) -> pd.DataFrame:
        """Extract first matching proteins for each TCR.

        Args:
            df: Processed DataFrame.
            top_k: Number of top matches to consider.

        Returns:
            pd.DataFrame: Long-form DataFrame with matching protein information.
        """
        # Get matching rows
        conditions = [df[f'match_{i}'] == 1 for i in range(top_k)]
        df_matched = df[pd.concat(conditions, axis=1).any(axis=1)]

        # Get first matching protein
        df_matched['first_matching_protein'] = None
        for i in range(top_k):
            mask = (df_matched['first_matching_protein'].isna()) & \
                   (df_matched[f'match_{i}'] == 1)
            df_matched.loc[mask, 'first_matching_protein'] = \
                df_matched.loc[mask, f'ref_protein_{i}'].apply(
                    self.protein_standardizer.standardize
                )

        # Create long-form DataFrame
        df_long = df_matched[[
            'clonotype_size',
            'pattern',
            'celltype',
            'first_matching_protein'
        ]].copy()

        df_long = df_long.rename(
            columns={'first_matching_protein': 'ref_protein'}
        )

        return df_long.dropna(subset=['ref_protein'])

    def _validate_inputs(self):
        """Validate input parameters."""
        if not self.condition_patterns:
            raise ValueError("condition_patterns cannot be empty")

        if not self.pattern_order:
            raise ValueError("pattern_order cannot be empty")

        if not self.protein_colors:
            raise ValueError("protein_colors cannot be empty")

        if not all(pattern in self.condition_patterns.values()
                  for pattern in self.pattern_order):
            raise ValueError("pattern_order must contain valid pattern names")

    def analyze(
        self,
        adata,
        top_k: int = 1,
        consider_cell_type: bool = False,
        top_n_proteins: int = 20,
        condition_column: str = "condition"
    ) -> pd.DataFrame:
        """Analyze antigen distribution across conditions.

        Args:
            adata: AnnData object containing the data.
            top_k: Number of top matches to consider.
            consider_cell_type: Whether to stratify by cell type.
            top_n_proteins: Number of top proteins to include in visualization.
            condition_column: Column name containing condition values.

        Returns:
            pd.DataFrame: Processed data with antigen distributions.
        """
        # Process data
        df = self._preprocess_data(adata.obs, condition_column)

        # Get first matching proteins
        df_long = self._get_matching_proteins(df, top_k)

        # Create visualizations
        self._create_visualizations(
            df_long,
            top_k,
            consider_cell_type,
            top_n_proteins
        )

        return df_long

    def _preprocess_data(
        self,
        df: pd.DataFrame,
        condition_column: str
    ) -> pd.DataFrame:
        """Preprocess the input data.

        Args:
            df: Input DataFrame.
            condition_column: Column name containing condition values.

        Returns:
            pd.DataFrame: Processed DataFrame with standardized columns.
        """
        df = df.copy()

        # Map conditions to patterns
        df['pattern'] = df[condition_column].map(self.condition_patterns)

        # Map cell types if mapping exists
        if self.cell_type_mapping:
            df['celltype'] = df[self.cell_type_column].map(self.cell_type_mapping)
        else:
            df['celltype'] = df[self.cell_type_column]

        return df

    def _normalize_distribution(
        self,
        data: pd.DataFrame,
        consider_cell_type: bool = False
    ) -> pd.DataFrame:
        """Normalize protein distribution within patterns.

        Args:
            data: Long-form DataFrame with protein frequencies.
            consider_cell_type: Whether to stratify by cell type.

        Returns:
            pd.DataFrame: Normalized frequency data.
        """
        # Group by relevant columns to get counts
        group_cols = ['ref_protein', 'pattern', 'celltype'] if consider_cell_type else ['ref_protein', 'pattern']
        counts = data.groupby(group_cols)['clonotype_size'].count().reset_index()

        # Calculate percentages within each pattern (and celltype if considered)
        norm_cols = ['pattern', 'celltype'] if consider_cell_type else ['pattern']
        totals = counts.groupby(norm_cols)['clonotype_size'].transform('sum')
        counts['normalized_value'] = (counts['clonotype_size'] / totals) * 100
        return counts

    def _create_visualizations(
        self,
        data: pd.DataFrame,
        top_k: int,
        consider_cell_type: bool,
        top_n_proteins: int
    ):
        """Create protein distribution visualizations.

        Args:
            data: Processed long-form DataFrame.
            top_k: Number of top matches considered.
            consider_cell_type: Whether to stratify by cell type.
            top_n_proteins: Number of top proteins to visualize.
        """
        # Create plots for expanded and non-expanded clones
        for clone_type, condition in [
            ('CE', 'clonotype_size > 1'),
            ('NE', 'clonotype_size == 1')
        ]:
            subset = data.query(condition)
            self._create_distribution_plot(
                subset,
                f"Protein Distribution - {clone_type}",
                f"protein_ratio_{clone_type}_k{top_k}",
                consider_cell_type,
                top_n_proteins
            )

    def _create_distribution_plot(
        self,
        data: pd.DataFrame,
        title: str,
        filename: str,
        consider_cell_type: bool,
        top_n_proteins: int
    ):
        """Create a stacked bar plot for protein distribution."""
        # Normalize data
        norm_data = self._normalize_distribution(data, consider_cell_type)

        # Get pattern counts
        pattern_counts = data.groupby('pattern').size()

        # Get top proteins
        if top_n_proteins:
            top_proteins = (norm_data.groupby('ref_protein')['clonotype_size']
                           .sum()
                           .sort_values(ascending=False)
                           .head(top_n_proteins)
                           .index)
            norm_data = norm_data[norm_data['ref_protein'].isin(top_proteins)]

        # Create pivot table
        if consider_cell_type:
            pivot = pd.pivot_table(norm_data,
                                 values='normalized_value',
                                 index=['pattern', 'celltype'],
                                 columns='ref_protein')
        else:
            pivot = pd.pivot_table(norm_data,
                                 values='normalized_value',
                                 index='pattern',
                                 columns='ref_protein')
        pivot = pivot.fillna(0)

        # Sort by pattern order
        pivot = pivot.reindex(self.pattern_order)

        # Create plot
        plt.figure(figsize=(12, 6))
        ax = plt.gca()

        # Plot bars
        bottom = np.zeros(len(pivot))
        for protein in pivot.columns:
            values = pivot[protein].values
            ax.bar(pivot.index, values, bottom=bottom,
                   label=protein, color=self.protein_colors.get(protein, '#808080'))
            bottom += values

        # Customize plot
        plt.title(title, pad=20)
        x_labels = []
        for pattern in pivot.index:
            try:
                x_labels.append(f'{pattern}\n(N={pattern_counts[pattern]})')
            except:
                x_labels.append(f'{pattern}\n(N=0)')
        plt.xticks(range(len(pivot.index)), x_labels)

        plt.xlabel('Disease Severity')
        plt.ylabel('Percentage (%)')

        # Add legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                  borderaxespad=0., title='Target Proteins')

        # Add percentage labels on bars
        for idx, severity in enumerate(pivot.index):
            total = 0
            for protein in pivot.columns:
                value = pivot.loc[severity, protein]
                if value > 0:  # Only add label if value is significant
                    plt.text(idx, total + value/2, f'{value:.1f}%',
                            ha='center', va='center')
                total += value

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"{filename}.pdf",
            bbox_inches='tight',
            dpi=300
        )
        plt.close()

    def _normalize_distribution(
        self,
        data: pd.DataFrame,
        consider_cell_type: bool = False
    ) -> pd.DataFrame:
        """Normalize protein distribution within patterns."""
        # Group by relevant columns to get counts
        group_cols = ['ref_protein', 'pattern', 'celltype'] if consider_cell_type else ['ref_protein', 'pattern']
        counts = data.groupby(group_cols)['clonotype_size'].count().reset_index()

        # Calculate percentages within each pattern (and celltype if considered)
        norm_cols = ['pattern', 'celltype'] if consider_cell_type else ['pattern']
        totals = counts.groupby(norm_cols)['clonotype_size'].transform('sum')
        counts['normalized_value'] = (counts['clonotype_size'] / totals) * 100
        return counts


