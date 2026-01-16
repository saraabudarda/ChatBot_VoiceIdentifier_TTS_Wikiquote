"""
Visualization utilities for Deezer Mood Detection Dataset
Professional, publication-ready plots with consistent styling
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from utils.config import COLORS, FIGURE_SIZE, FONT_SIZE, TITLE_SIZE, EMOTIONAL_QUADRANTS


# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_distribution_histogram(
    df: pd.DataFrame,
    column: str,
    split_name: str = None,
    bins: int = 50,
    color: str = None,
    ax: plt.Axes = None
) -> plt.Figure:
    """
    Plot histogram for a single variable.
    
    Args:
        df: DataFrame containing the data
        column: Column name to plot
        split_name: Name of the split (for title)
        bins: Number of bins
        color: Bar color
        ax: Matplotlib axes object (optional)
        
    Returns:
        Matplotlib figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    else:
        fig = ax.get_figure()
    
    if color is None:
        color = COLORS['primary']
    
    ax.hist(df[column], bins=bins, color=color, alpha=0.7, edgecolor='black')
    
    title = f'Distribution of {column.capitalize()}'
    if split_name:
        title += f' ({split_name.capitalize()} Set)'
    
    ax.set_xlabel(column.capitalize(), fontsize=FONT_SIZE)
    ax.set_ylabel('Frequency', fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_val = df[column].mean()
    std_val = df[column].std()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_overlaid_distributions(
    datasets: Dict[str, pd.DataFrame],
    column: str,
    bins: int = 50
) -> plt.Figure:
    """
    Plot overlaid histograms for multiple dataset splits.
    
    Args:
        datasets: Dictionary of DataFrames with split names as keys
        column: Column name to plot
        bins: Number of bins
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    for split_name, df in datasets.items():
        color = COLORS.get(split_name, COLORS['primary'])
        ax.hist(
            df[column],
            bins=bins,
            alpha=0.5,
            label=split_name.capitalize(),
            color=color,
            edgecolor='black'
        )
    
    ax.set_xlabel(column.capitalize(), fontsize=FONT_SIZE)
    ax.set_ylabel('Frequency', fontsize=FONT_SIZE)
    ax.set_title(f'{column.capitalize()} Distribution Across Splits', fontsize=TITLE_SIZE, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_density_comparison(
    datasets: Dict[str, pd.DataFrame],
    column: str
) -> plt.Figure:
    """
    Plot kernel density estimation for multiple splits.
    
    Args:
        datasets: Dictionary of DataFrames with split names as keys
        column: Column name to plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    for split_name, df in datasets.items():
        color = COLORS.get(split_name, COLORS['primary'])
        df[column].plot.kde(
            ax=ax,
            label=split_name.capitalize(),
            color=color,
            linewidth=2
        )
    
    ax.set_xlabel(column.capitalize(), fontsize=FONT_SIZE)
    ax.set_ylabel('Density', fontsize=FONT_SIZE)
    ax.set_title(f'{column.capitalize()} Density Comparison', fontsize=TITLE_SIZE, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_boxplot(
    datasets: Dict[str, pd.DataFrame],
    column: str
) -> plt.Figure:
    """
    Create boxplot for comparing distributions across splits.
    
    Args:
        datasets: Dictionary of DataFrames with split names as keys
        column: Column name to plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Prepare data
    data_list = []
    labels = []
    for split_name, df in datasets.items():
        data_list.append(df[column].dropna())
        labels.append(split_name.capitalize())
    
    # Create boxplot
    bp = ax.boxplot(data_list, labels=labels, patch_artist=True, showmeans=True)
    
    # Color boxes
    for patch, split_name in zip(bp['boxes'], datasets.keys()):
        color = COLORS.get(split_name, COLORS['primary'])
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel(column.capitalize(), fontsize=FONT_SIZE)
    ax.set_title(f'{column.capitalize()} Distribution by Split', fontsize=TITLE_SIZE, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_scatter_valence_arousal(
    df: pd.DataFrame,
    split_name: str = None,
    sample_size: int = None,
    alpha: float = 0.3
) -> plt.Figure:
    """
    Create scatter plot of valence vs arousal with quadrant lines.
    
    Args:
        df: DataFrame containing valence and arousal
        split_name: Name of the split (for title)
        sample_size: Number of points to sample (for large datasets)
        alpha: Point transparency
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Sample if dataset is large
    if sample_size and len(df) > sample_size:
        df_plot = df.sample(n=sample_size, random_state=42)
    else:
        df_plot = df
    
    # Create scatter plot
    scatter = ax.scatter(
        df_plot['valence'],
        df_plot['arousal'],
        alpha=alpha,
        c=COLORS['primary'],
        s=20,
        edgecolors='none'
    )
    
    # Add quadrant lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Add quadrant labels
    quadrant_labels = {
        'Q1': 'Happy/Excited\n(+V, +A)',
        'Q2': 'Angry/Tense\n(-V, +A)',
        'Q3': 'Sad/Depressed\n(-V, -A)',
        'Q4': 'Calm/Relaxed\n(+V, -A)'
    }
    
    # Get axis limits
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    
    # Q1: top-right
    ax.text(x_lim[1] * 0.7, y_lim[1] * 0.85, quadrant_labels['Q1'],
            fontsize=10, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    # Q2: top-left
    ax.text(x_lim[0] * 0.7, y_lim[1] * 0.85, quadrant_labels['Q2'],
            fontsize=10, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    # Q3: bottom-left
    ax.text(x_lim[0] * 0.7, y_lim[0] * 0.85, quadrant_labels['Q3'],
            fontsize=10, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    # Q4: bottom-right
    ax.text(x_lim[1] * 0.7, y_lim[0] * 0.85, quadrant_labels['Q4'],
            fontsize=10, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Valence (Negativity ← → Positivity)', fontsize=FONT_SIZE)
    ax.set_ylabel('Arousal (Low Energy ← → High Energy)', fontsize=FONT_SIZE)
    
    title = 'Valence-Arousal Emotional Space'
    if split_name:
        title += f' ({split_name.capitalize()} Set)'
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Create correlation heatmap for numerical variables.
    
    Args:
        df: DataFrame containing the data
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Compute correlation matrix
    corr_matrix = df[['valence', 'arousal']].corr()
    
    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title('Correlation Matrix: Valence vs Arousal', fontsize=TITLE_SIZE, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_hexbin_density(
    df: pd.DataFrame,
    split_name: str = None,
    gridsize: int = 30
) -> plt.Figure:
    """
    Create hexbin plot for high-density visualization of valence-arousal space.
    
    Args:
        df: DataFrame containing valence and arousal
        split_name: Name of the split (for title)
        gridsize: Number of hexagons in x-direction
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    hexbin = ax.hexbin(
        df['valence'],
        df['arousal'],
        gridsize=gridsize,
        cmap='YlOrRd',
        mincnt=1
    )
    
    # Add quadrant lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Valence (Negativity ← → Positivity)', fontsize=FONT_SIZE)
    ax.set_ylabel('Arousal (Low Energy ← → High Energy)', fontsize=FONT_SIZE)
    
    title = 'Valence-Arousal Density Plot'
    if split_name:
        title += f' ({split_name.capitalize()} Set)'
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
    
    cb = plt.colorbar(hexbin, ax=ax)
    cb.set_label('Count', fontsize=FONT_SIZE)
    
    plt.tight_layout()
    return fig


def plot_quadrant_distribution(quadrant_df: pd.DataFrame) -> plt.Figure:
    """
    Create bar plot of emotional quadrant distribution.
    
    Args:
        quadrant_df: DataFrame with quadrant statistics
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']  # Green, Red, Blue, Orange
    
    bars = ax.bar(
        quadrant_df['Quadrant'],
        quadrant_df['Count'],
        color=colors,
        alpha=0.7,
        edgecolor='black'
    )
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, quadrant_df['Percentage']):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{pct:.1f}%',
            ha='center',
            va='bottom',
            fontsize=FONT_SIZE
        )
    
    ax.set_xlabel('Emotional Quadrant', fontsize=FONT_SIZE)
    ax.set_ylabel('Number of Tracks', fontsize=FONT_SIZE)
    ax.set_title('Distribution Across Emotional Quadrants', fontsize=TITLE_SIZE, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig
