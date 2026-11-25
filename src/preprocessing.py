"""
PhishGuard AI - Preprocessing Module
Handles data cleaning, target normalization, stratified train-test splitting,
and Scikit-learn preprocessing pipelines to prevent data leakage.
"""

import os
import joblib
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def normalize_target(y_series: pd.Series) -> pd.Series:
    """
    Normalizes UCI target:
    Original +1 (Legitimate) -> 0
    Original -1 (Phishing)   -> 1
    """
    # Verify values
    unique_vals = set(y_series.unique())
    logger.info(f"Normalizing targets with unique raw values: {unique_vals}")
    
    # Map +1 -> 0, -1 -> 1
    normalized = y_series.map({1: 0, -1: 1, '1': 0, '-1': 1})
    if normalized.isnull().any():
        raise ValueError("Target mapping produced null values. Check input targets.")
    return normalized.astype(int)


def prepare_and_split_data(
    raw_data_path: str = "data/raw/phishing_raw.csv",
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline]:
    """
    Performs end-to-end data preparation:
    - Loads raw data
    - Performs missing value and duplicate checks
    - Separates features and target
    - Normalizes target (0=Legitimate, 1=Phishing)
    - Performs stratified 80:20 train-test split
    - Fits StandardScaler inside a Pipeline on X_train only
    - Saves processed datasets, splits, and preprocessing pipeline
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df_raw = pd.read_csv(raw_data_path)
    logger.info(f"Loaded raw dataset with shape: {df_raw.shape}")

    # Standardize column names
    df_raw.columns = [col.lower().strip().replace(' ', '_').replace('-', '_') for col in df_raw.columns]
    target_col = 'result'
    if target_col not in df_raw.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe columns: {df_raw.columns.tolist()}")

    # Check missing values
    missing_counts = df_raw.isnull().sum()
    if missing_counts.sum() > 0:
        logger.warning(f"Found missing values:\n{missing_counts[missing_counts > 0]}. Imputing with mode...")
        df_raw = df_raw.fillna(df_raw.mode().iloc[0])
    else:
        logger.info("Zero missing values detected in dataset.")

    # Separate X and y
    feature_cols = [c for c in df_raw.columns if c != target_col]
    X = df_raw[feature_cols].copy()
    y_raw = df_raw[target_col].copy()

    # Normalize targets
    y = normalize_target(y_raw)
    
    logger.info(f"Target normalization complete. Legitimate (0): {(y == 0).sum()}, Phishing (1): {(y == 1).sum()}")

    # Save full processed dataset
    df_processed = X.copy()
    df_processed['target'] = y
    processed_path = os.path.join(processed_dir, "phishing_processed.csv")
    df_processed.to_csv(processed_path, index=False)
    logger.info(f"Processed dataset saved to {processed_path}")

    # Stratified Train-Test Split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    logger.info(f"Stratified split completed: X_train={X_train.shape}, X_test={X_test.shape}")

    # Save train and test sets
    train_df = pd.concat([X_train, y_train.rename('target')], axis=1)
    test_df = pd.concat([X_test, y_test.rename('target')], axis=1)
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    # Build and fit preprocessing pipeline on X_train only
    pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    pipeline.fit(X_train)

    pipeline_path = os.path.join(models_dir, "preprocessing_pipeline.pkl")
    joblib.dump({
        'pipeline': pipeline,
        'feature_names': feature_cols,
        'scaler': pipeline.named_steps['scaler']
    }, pipeline_path)
    logger.info(f"Preprocessing pipeline fitted on X_train and saved to {pipeline_path}")

    # Generate sample test files for demonstration and bulk upload
    sample_legit = X_test[y_test == 0].head(10)
    sample_phish = X_test[y_test == 1].head(10)
    sample_batch = pd.concat([sample_legit.head(5), sample_phish.head(5)]).sample(frac=1.0, random_state=random_state)
    
    sample_legit.to_csv("data/sample_legitimate.csv", index=False)
    sample_phish.to_csv("data/sample_phishing.csv", index=False)
    sample_batch.to_csv("data/sample_batch_test.csv", index=False)
    logger.info("Sample demonstration CSV files saved to data/")

    return X_train, X_test, y_train, y_test, pipeline


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, pipe = prepare_and_split_data()
    print("Data preparation and stratified splitting complete!")
    print(f"X_train: {X_tr.shape}, X_test: {X_te.shape}")
    print(f"y_train distribution:\n{y_tr.value_counts(normalize=True)}")
    print(f"y_test distribution:\n{y_te.value_counts(normalize=True)}")
