"""
Tests for Inference, Batch Prediction, Schema Validation & Thresholds
"""

import os
import pytest
import pandas as pd
from src.prediction import PhishGuardPredictor, EXPECTED_FEATURES


@pytest.fixture
def predictor():
    return PhishGuardPredictor()


def test_single_prediction_valid(predictor):
    sample = {f: 1 for f in EXPECTED_FEATURES}
    res = predictor.predict_single(sample)
    assert 'is_phishing' in res
    assert 'prediction_label' in res
    assert 'confidence' in res
    assert 'phishing_probability' in res
    assert isinstance(res['is_phishing'], (bool, int))
    assert 0.0 <= res['confidence'] <= 1.0


def test_schema_validation_rejection(predictor):
    invalid_sample = {'having_ip_address': 1}
    with pytest.raises(ValueError):
        predictor.predict_single(invalid_sample)


def test_batch_prediction_sample_csv(predictor):
    sample_batch_path = "data/sample_batch_test.csv"
    assert os.path.exists(sample_batch_path)
    df = pd.read_csv(sample_batch_path)
    res_df, summary = predictor.predict_batch(df)
    assert len(res_df) == len(df)
    assert 'predicted_class' in res_df.columns
    assert 'prediction_label' in res_df.columns
    assert summary['total_analyzed'] == len(df)


def test_threshold_simulation(predictor):
    sim = predictor.simulate_threshold_experiment(threshold=0.35)
    assert 0.0 <= sim['accuracy'] <= 1.0
    assert 0.0 <= sim['recall'] <= 1.0
    assert 0.0 <= sim['precision'] <= 1.0
    assert sim['tp'] + sim['fp'] + sim['fn'] + sim['tn'] == sim['total_samples']
