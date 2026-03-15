# src/data_utils.py

from pathlib import Path
import pandas as pd

from src.config import TARGET_COL, ID_COLS, FEATURE_COLS


def load_data(csv_path: str | Path) -> pd.DataFrame:
    '''
    Load data from a CSV file and return a DataFrame.
    '''
    df = pd.read_csv(csv_path)
    return df


def get_feature_target(df: pd.DataFrame):
    '''
    Return X, y, and ID dataframe.'''
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    ids = df[ID_COLS].copy()
    return X, y, ids


def basic_data_check(df: pd.DataFrame):
    '''
    Simple dataset diagnostics
    '''
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nTarget distribution:")
    print(df[TARGET_COL].value_counts(dropna=False))
    print("\nMissing values:")
    print((df.isnull().mean() * 100).sort_values(ascending=False))
