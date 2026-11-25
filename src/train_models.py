"""
PhishGuard AI - Model Training Module
Trains Logistic Regression, Decision Tree, and MLP Neural Network.
Measures training time, prediction time, and saves trained model artifacts.
"""

import os
import time
import json
import logging
import joblib
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_and_save_models(
    train_path: str = "data/processed/train.csv",
    test_path: str = "data/processed/test.csv",
    pipeline_path: str = "models/preprocessing_pipeline.pkl",
    models_dir: str = "models",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains:
    1. Logistic Regression
    2. Decision Tree Classifier
    3. Multi-Layer Perceptron (MLP) Neural Network

    Saves all models, evaluation metrics, and identifies the best model
    giving special importance to phishing recall while maintaining high precision.
    """
    os.makedirs(models_dir, exist_ok=True)

    # Load splits
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    pipe_dict = joblib.load(pipeline_path)
    scaler = pipe_dict['scaler']

    feature_cols = [c for c in train_df.columns if c != 'target']
    X_train_raw = train_df[feature_cols]
    y_train = train_df['target']
    X_test_raw = test_df[feature_cols]
    y_test = test_df['target']

    # Scale features for models that require normalization (Logistic Regression & MLP)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train_raw), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=feature_cols)

    # Models definition
    models_config = {
        'Logistic Regression': {
            'model': LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver='lbfgs',
                random_state=random_state
            ),
            'use_scaled': True,
            'filename': 'logistic_regression.pkl',
            'desc': 'Linear probabilistic classifier with L2 regularization.'
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(
                criterion='gini',
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state
            ),
            'use_scaled': False,  # Tree-based model is invariant to monotonic scaling
            'filename': 'decision_tree.pkl',
            'desc': 'Non-linear tree-based classifier with rule explainability.'
        },
        'MLP Classifier': {
            'model': MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate_init=0.001,
                max_iter=350,
                early_stopping=True,
                n_iter_no_change=15,
                validation_fraction=0.1,
                random_state=random_state
            ),
            'use_scaled': True,
            'filename': 'mlp_classifier.pkl',
            'desc': 'Feedforward deep artificial neural network with early stopping.'
        }
    }

    results = {}

    for name, cfg in models_config.items():
        logger.info(f"--- Training {name} ---")
        model = cfg['model']
        X_tr = X_train_scaled if cfg['use_scaled'] else X_train_raw
        X_te = X_test_scaled if cfg['use_scaled'] else X_test_raw

        # Measure training time
        t0 = time.perf_counter()
        model.fit(X_tr, y_train)
        train_time = time.perf_counter() - t0

        # Measure prediction time
        t1 = time.perf_counter()
        y_pred = model.predict(X_te)
        pred_time = time.perf_counter() - t1
        y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else None

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        logger.info(f"{name} Results -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, Train Time: {train_time:.4f}s")

        model_artifact = {
            'name': name,
            'model': model,
            'use_scaled': cfg['use_scaled'],
            'feature_names': feature_cols,
            'train_time': train_time,
            'pred_time': pred_time,
            'metrics': {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc),
                'train_time_sec': float(train_time),
                'pred_time_sec': float(pred_time),
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            },
            'description': cfg['desc']
        }

        # If neural net, record loss curve
        if hasattr(model, 'loss_curve_'):
            model_artifact['loss_curve'] = [float(l) for l in model.loss_curve_]

        save_path = os.path.join(models_dir, cfg['filename'])
        joblib.dump(model_artifact, save_path)
        logger.info(f"Saved model artifact to {save_path}")

        results[name] = model_artifact

    # Best model selection:
    # In cybersecurity phishing detection, catching phishing attacks (Recall) is mission-critical
    # while controlling False Positives (F1-score / ROC-AUC). We use a composite score:
    # 0.4 * Recall + 0.3 * F1 + 0.3 * ROC_AUC
    best_name = None
    best_score = -1.0
    for name, art in results.items():
        m = art['metrics']
        composite_score = 0.4 * m['recall'] + 0.3 * m['f1_score'] + 0.3 * m['roc_auc']
        logger.info(f"Model {name} Evaluation Composite Score: {composite_score:.4f}")
        if composite_score > best_score:
            best_score = composite_score
            best_name = name

    logger.info(f"Selected Best Performing Model: {best_name} with Composite Score: {best_score:.4f}")
    best_model_artifact = results[best_name].copy()
    best_model_artifact['is_best'] = True
    best_save_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model_artifact, best_save_path)
    logger.info(f"Saved best model artifact to {best_save_path}")

    return results


if __name__ == "__main__":
    trained_results = train_and_save_models()
    print("All models trained and saved successfully!")
