"""
Statistical analysis utilities for Deezer Mood Detection Dataset
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from utils.config import TARGET_COLUMNS


def compute_descriptive_stats(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Compute comprehensive descriptive statistics for specified columns.
    
    Args:
        df: DataFrame containing the data
        columns: List of column names to analyze (default: TARGET_COLUMNS)
        
    Returns:
        DataFrame with descriptive statistics
    """
    if columns is None:
        columns = TARGET_COLUMNS
    
    stats_dict = {}
    
    for col in columns:
        if col in df.columns:
            stats_dict[col] = {
                'Count': df[col].count(),
                'Mean': df[col].mean(),
                'Std': df[col].std(),
                'Min': df[col].min(),
                '25%': df[col].quantile(0.25),
                'Median': df[col].median(),
                '75%': df[col].quantile(0.75),
                'Max': df[col].max(),
                'Skewness': df[col].skew(),
                'Kurtosis': df[col].kurtosis(),
            }
    
    stats_df = pd.DataFrame(stats_dict).T
    return stats_df


def compute_correlation(df: pd.DataFrame, method: str = 'pearson') -> Dict[str, float]:
    """
    Compute correlation between valence and arousal.
    
    Args:
        df: DataFrame containing valence and arousal columns
        method: Correlation method ('pearson' or 'spearman')
        
    Returns:
        Dictionary with correlation coefficient and p-value
    """
    if method == 'pearson':
        corr, p_value = stats.pearsonr(df['valence'], df['arousal'])
    elif method == 'spearman':
        corr, p_value = stats.spearmanr(df['valence'], df['arousal'])
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    return {
        'correlation': corr,
        'p_value': p_value,
        'method': method,
        'significant': p_value < 0.05
    }


def compare_distributions(df1: pd.DataFrame, df2: pd.DataFrame, column: str) -> Dict[str, any]:
    """
    Compare distributions between two datasets using statistical tests.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        column: Column name to compare
        
    Returns:
        Dictionary with test results
    """
    # Kolmogorov-Smirnov test
    ks_statistic, ks_pvalue = stats.ks_2samp(df1[column], df2[column])
    
    # Mann-Whitney U test (non-parametric)
    mw_statistic, mw_pvalue = stats.mannwhitneyu(df1[column], df2[column])
    
    # T-test (parametric)
    t_statistic, t_pvalue = stats.ttest_ind(df1[column], df2[column])
    
    return {
        'ks_test': {
            'statistic': ks_statistic,
            'p_value': ks_pvalue,
            'significant': ks_pvalue < 0.05
        },
        'mann_whitney': {
            'statistic': mw_statistic,
            'p_value': mw_pvalue,
            'significant': mw_pvalue < 0.05
        },
        't_test': {
            'statistic': t_statistic,
            'p_value': t_pvalue,
            'significant': t_pvalue < 0.05
        }
    }


def test_normality(df: pd.DataFrame, column: str) -> Dict[str, any]:
    """
    Test if a distribution is normal using multiple tests.
    
    Args:
        df: DataFrame containing the data
        column: Column name to test
        
    Returns:
        Dictionary with normality test results
    """
    # Shapiro-Wilk test (good for small samples)
    if len(df) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(df[column].dropna())
    else:
        shapiro_stat, shapiro_p = None, None
    
    # Anderson-Darling test
    anderson_result = stats.anderson(df[column].dropna())
    
    # Kolmogorov-Smirnov test against normal distribution
    ks_stat, ks_p = stats.kstest(
        df[column].dropna(),
        'norm',
        args=(df[column].mean(), df[column].std())
    )
    
    return {
        'shapiro_wilk': {
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'normal': shapiro_p > 0.05 if shapiro_p is not None else None
        } if shapiro_stat is not None else None,
        'anderson_darling': {
            'statistic': anderson_result.statistic,
            'critical_values': anderson_result.critical_values.tolist(),
            'significance_levels': anderson_result.significance_level.tolist()
        },
        'ks_test': {
            'statistic': ks_stat,
            'p_value': ks_p,
            'normal': ks_p > 0.05
        }
    }


def compute_split_statistics(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute statistics for each dataset split.
    
    Args:
        datasets: Dictionary of DataFrames with split names as keys
        
    Returns:
        DataFrame with statistics for each split
    """
    all_stats = []
    
    for split_name, df in datasets.items():
        for col in TARGET_COLUMNS:
            if col in df.columns:
                all_stats.append({
                    'Split': split_name.capitalize(),
                    'Variable': col.capitalize(),
                    'Count': df[col].count(),
                    'Mean': f"{df[col].mean():.4f}",
                    'Std': f"{df[col].std():.4f}",
                    'Min': f"{df[col].min():.4f}",
                    'Max': f"{df[col].max():.4f}",
                    'Skewness': f"{df[col].skew():.4f}",
                    'Kurtosis': f"{df[col].kurtosis():.4f}",
                })
    
    return pd.DataFrame(all_stats)


def identify_emotional_quadrant(valence: float, arousal: float) -> str:
    """
    Identify which emotional quadrant a point belongs to.
    
    Args:
        valence: Valence value
        arousal: Arousal value
        
    Returns:
        Quadrant name (Q1, Q2, Q3, or Q4)
    """
    if valence >= 0 and arousal >= 0:
        return 'Q1'  # Happy/Excited
    elif valence < 0 and arousal >= 0:
        return 'Q2'  # Angry/Tense
    elif valence < 0 and arousal < 0:
        return 'Q3'  # Sad/Depressed
    else:
        return 'Q4'  # Calm/Relaxed


def compute_quadrant_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute distribution of samples across emotional quadrants.
    
    Args:
        df: DataFrame with valence and arousal columns
        
    Returns:
        DataFrame with quadrant counts and percentages
    """
    df['quadrant'] = df.apply(
        lambda row: identify_emotional_quadrant(row['valence'], row['arousal']),
        axis=1
    )
    
    quadrant_counts = df['quadrant'].value_counts().sort_index()
    quadrant_pct = (quadrant_counts / len(df) * 100).round(2)
    
    quadrant_names = {
        'Q1': 'Happy/Excited',
        'Q2': 'Angry/Tense',
        'Q3': 'Sad/Depressed',
        'Q4': 'Calm/Relaxed'
    }
    
    result = pd.DataFrame({
        'Quadrant': [quadrant_names[q] for q in quadrant_counts.index],
        'Count': quadrant_counts.values,
        'Percentage': quadrant_pct.values
    })
    
    return result
