"""
Data loading and validation utilities for Deezer Mood Detection Dataset
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import os
from utils.config import (
    DATA_DIR, TRAIN_FILE, VALIDATION_FILE, TEST_FILE,
    ALL_COLUMNS, TARGET_COLUMNS, ID_COLUMNS, METADATA_COLUMNS
)


def load_dataset(file_name: str) -> pd.DataFrame:
    """
    Load a single CSV file from the dataset directory.
    
    Args:
        file_name: Name of the CSV file to load
        
    Returns:
        DataFrame containing the loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    file_path = os.path.join(DATA_DIR, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    return df


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    """
    Load all three dataset splits (train, validation, test).
    
    Returns:
        Dictionary with keys 'train', 'validation', 'test' containing DataFrames
    """
    datasets = {
        'train': load_dataset(TRAIN_FILE),
        'validation': load_dataset(VALIDATION_FILE),
        'test': load_dataset(TEST_FILE)
    }
    
    return datasets


def validate_dataset(df: pd.DataFrame, split_name: str) -> Dict[str, any]:
    """
    Validate a dataset and return quality metrics.
    
    Args:
        df: DataFrame to validate
        split_name: Name of the split (for reporting)
        
    Returns:
        Dictionary containing validation results
    """
    validation_results = {
        'split_name': split_name,
        'n_samples': len(df),
        'n_features': len(df.columns),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'total_missing': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
    }
    
    # Check for target columns
    if all(col in df.columns for col in TARGET_COLUMNS):
        validation_results['valence_range'] = (df['valence'].min(), df['valence'].max())
        validation_results['arousal_range'] = (df['arousal'].min(), df['arousal'].max())
        validation_results['valence_nulls'] = df['valence'].isnull().sum()
        validation_results['arousal_nulls'] = df['arousal'].isnull().sum()
    
    return validation_results


def combine_datasets(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine all dataset splits into a single DataFrame with split identifier.
    
    Args:
        datasets: Dictionary of DataFrames with split names as keys
        
    Returns:
        Combined DataFrame with 'split' column added
    """
    combined_dfs = []
    
    for split_name, df in datasets.items():
        df_copy = df.copy()
        df_copy['split'] = split_name
        combined_dfs.append(df_copy)
    
    combined = pd.concat(combined_dfs, ignore_index=True)
    return combined


def get_dataset_summary(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a summary table of all dataset splits.
    
    Args:
        datasets: Dictionary of DataFrames with split names as keys
        
    Returns:
        DataFrame with summary statistics for each split
    """
    summary_data = []
    
    for split_name, df in datasets.items():
        summary_data.append({
            'Split': split_name.capitalize(),
            'Samples': len(df),
            'Features': len(df.columns),
            'Missing Values': df.isnull().sum().sum(),
            'Duplicates': df.duplicated().sum(),
        })
    
    # Add combined row
    combined = combine_datasets(datasets)
    summary_data.append({
        'Split': 'Combined',
        'Samples': len(combined),
        'Features': len(combined.columns) - 1,  # Exclude 'split' column
        'Missing Values': combined.drop('split', axis=1).isnull().sum().sum(),
        'Duplicates': combined.drop('split', axis=1).duplicated().sum(),
    })
    
    return pd.DataFrame(summary_data)


def check_data_consistency(datasets: Dict[str, pd.DataFrame]) -> Dict[str, any]:
    """
    Check consistency across dataset splits.
    
    Args:
        datasets: Dictionary of DataFrames with split names as keys
        
    Returns:
        Dictionary with consistency check results
    """
    # Get column names from each split
    columns_by_split = {name: set(df.columns) for name, df in datasets.items()}
    
    # Check if all splits have the same columns
    all_columns = [set(df.columns) for df in datasets.values()]
    columns_consistent = all(cols == all_columns[0] for cols in all_columns)
    
    # Check data types consistency
    dtypes_by_split = {name: df.dtypes.to_dict() for name, df in datasets.items()}
    
    consistency_results = {
        'columns_consistent': columns_consistent,
        'columns_by_split': {k: list(v) for k, v in columns_by_split.items()},
        'dtypes_consistent': all(
            dtypes_by_split['train'] == dtypes 
            for dtypes in dtypes_by_split.values()
        ),
        'dtypes_by_split': dtypes_by_split,
    }
    
    return consistency_results


def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> Tuple[pd.Series, int]:
    """
    Detect outliers using the IQR method.
    
    Args:
        df: DataFrame containing the data
        column: Column name to check for outliers
        multiplier: IQR multiplier (default 1.5 for standard outlier detection)
        
    Returns:
        Tuple of (boolean Series indicating outliers, count of outliers)
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    outlier_count = outliers.sum()
    
    return outliers, outlier_count
