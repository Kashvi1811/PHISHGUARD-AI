"""
Tests for Data Loader & Preprocessing Pipeline
"""

import os
import pytest
import pandas as pd
import numpy as np


def test_raw_data_exists():
    raw_path = "data/raw/phishing_raw.csv"
    assert os.path.exists(raw_path), f"Missing raw file: {raw_path}"
    df = pd.read_csv(raw_path)
    assert len(df) == 11055, f"Expected 11,055 rows, got {len(df)}"
    assert df.shape[1] == 31, f"Expected 31 columns, got {df.shape[1]}"


def test_processed_data_schema():
    proc_path = "data/processed/phishing_processed.csv"
    assert os.path.exists(proc_path), f"Missing processed file: {proc_path}"
    df = pd.read_csv(proc_path)
    assert len(df) == 11055
    assert 'target' in df.columns
    assert set(df['target'].unique()) == {0, 1}, f"Target labels must be {0, 1}, got {set(df['target'].unique())}"
    assert df.isnull().sum().sum() == 0, "Processed dataset must not contain NaN"


def test_train_test_splits():
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"
    assert os.path.exists(train_path) and os.path.exists(test_path)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    assert len(train_df) == 8844, f"Expected 8844 train rows (80%), got {len(train_df)}"
    assert len(test_df) == 2211, f"Expected 2211 test rows (20%), got {len(test_df)}"
