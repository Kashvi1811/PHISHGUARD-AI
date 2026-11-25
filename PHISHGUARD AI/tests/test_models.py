"""
Tests for Model Artifacts and Evaluation Performance
"""

import os
import pytest
import joblib
import pandas as pd


def test_model_artifacts_exist():
    required_models = [
        "logistic_regression.pkl",
        "decision_tree.pkl",
        "mlp_classifier.pkl",
        "best_model.pkl",
        "preprocessing_pipeline.pkl"
    ]
    for m in required_models:
        path = os.path.join("models", m)
        assert os.path.exists(path), f"Missing model file: {path}"


def test_model_loading_and_performance():
    test_df = pd.read_csv("data/processed/test.csv")
    pipe_dict = joblib.load("models/preprocessing_pipeline.pkl")
    scaler = pipe_dict['scaler']
    feature_cols = [c for c in test_df.columns if c != 'target']
    
    X_test_raw = test_df[feature_cols]
    y_test = test_df['target']
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=feature_cols)

    for m_file in ["logistic_regression.pkl", "decision_tree.pkl", "mlp_classifier.pkl", "best_model.pkl"]:
        art = joblib.load(os.path.join("models", m_file))
        model = art['model']
        use_scaled = art.get('use_scaled', True)
        X_eval = X_test_scaled if use_scaled else X_test_raw
        
        preds = model.predict(X_eval)
        acc = (preds == y_test).mean()
        assert acc >= 0.90, f"Model {m_file} failed benchmark with accuracy {acc:.4f} < 0.90"
