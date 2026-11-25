"""
PhishGuard AI - Prediction & Inference Engine
Provides single sample inference, bulk batch CSV prediction, input schema validation,
influential feature explanation, and ethical threshold sensitivity simulations.
"""

import os
import joblib
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_FEATURES = [
    'having_ip_address', 'url_length', 'shortining_service', 'having_at_symbol',
    'double_slash_redirecting', 'prefix_suffix', 'having_sub_domain', 'sslfinal_state',
    'domain_registration_length', 'favicon', 'port', 'https_token', 'request_url',
    'url_of_anchor', 'links_in_tags', 'sfh', 'submitting_to_email', 'abnormal_url',
    'redirect', 'on_mouseover', 'rightclick', 'popupwindow', 'iframe',
    'age_of_domain', 'dnsrecord', 'web_traffic', 'page_rank', 'google_index',
    'links_pointing_to_page', 'statistical_report'
]


class PhishGuardPredictor:
    """
    Inference and security intelligence engine for PhishGuard AI.
    Loads trained models and preprocessing pipelines with caching.
    """
    def __init__(self, models_dir: str = "models", pipeline_path: str = "models/preprocessing_pipeline.pkl"):
        self.models_dir = models_dir
        self.pipeline_dict = joblib.load(pipeline_path)
        self.scaler = self.pipeline_dict['scaler']
        self.feature_names = self.pipeline_dict['feature_names']
        self._models_cache = {}

    def get_model(self, model_name: str = "best_model.pkl") -> Dict[str, Any]:
        """Loads and caches model artifact dictionary."""
        if model_name not in self._models_cache:
            path = os.path.join(self.models_dir, model_name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            self._models_cache[model_name] = joblib.load(path)
        return self._models_cache[model_name]

    def validate_input_dataframe(self, df: pd.DataFrame) -> Tuple[bool, str, pd.DataFrame]:
        """
        Validates whether input dataframe contains all 30 required features.
        Standardizes column names to lowercase snake_case.
        """
        clean_df = df.copy()
        clean_df.columns = [c.lower().strip().replace(' ', '_').replace('-', '_') for c in clean_df.columns]
        
        # Exclude target column if present in input
        for target_candidate in ['result', 'target', 'class', 'label']:
            if target_candidate in clean_df.columns:
                clean_df = clean_df.drop(columns=[target_candidate])

        missing = [f for f in EXPECTED_FEATURES if f not in clean_df.columns]
        if missing:
            return False, f"Missing {len(missing)} required features: {missing[:5]}...", clean_df

        # Ensure correct column ordering
        clean_df = clean_df[EXPECTED_FEATURES]

        # Check values
        for col in clean_df.columns:
            non_numerics = pd.to_numeric(clean_df[col], errors='coerce')
            if non_numerics.isnull().any():
                return False, f"Feature '{col}' contains non-numeric entries.", clean_df
            clean_df[col] = non_numerics

        return True, "Validation successful.", clean_df

    def predict_single(
        self,
        feature_dict: Dict[str, Any],
        model_filename: str = "best_model.pkl",
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Predicts threat class for a single feature dictionary.
        Returns:
            - is_phishing (bool)
            - label ("Phishing Website" or "Legitimate Website")
            - confidence (probability of the predicted class)
            - phishing_probability (probability of class 1)
            - risk_level ("CRITICAL", "HIGH", "MODERATE", "LOW", "SAFE")
            - top_contributing_features
        """
        df_single = pd.DataFrame([feature_dict])
        is_valid, msg, clean_df = self.validate_input_dataframe(df_single)
        if not is_valid:
            raise ValueError(f"Input validation error: {msg}")

        artifact = self.get_model(model_filename)
        model = artifact['model']
        use_scaled = artifact.get('use_scaled', True)

        if use_scaled:
            X_input = pd.DataFrame(self.scaler.transform(clean_df), columns=self.feature_names)
        else:
            X_input = clean_df.copy()

        if hasattr(model, 'predict_proba'):
            prob_phishing = float(model.predict_proba(X_input)[0, 1])
        else:
            prob_phishing = float(model.predict(X_input)[0])

        is_phishing = prob_phishing >= threshold
        pred_label = "Phishing Website" if is_phishing else "Legitimate Website"
        confidence = prob_phishing if is_phishing else (1.0 - prob_phishing)

        # Risk Level
        if prob_phishing >= 0.85:
            risk = "CRITICAL THREAT"
            color = "#ef4444"
        elif prob_phishing >= 0.65:
            risk = "HIGH RISK"
            color = "#f97316"
        elif prob_phishing >= 0.45:
            risk = "SUSPICIOUS / ELEVATED"
            color = "#eab308"
        elif prob_phishing >= 0.20:
            risk = "LOW RISK"
            color = "#3b82f6"
        else:
            risk = "VERIFIED SAFE"
            color = "#10b981"

        # Explain top indicators for this instance
        # Inspect features with value = -1 (dangerous)
        dangerous_feats = [col for col in clean_df.columns if clean_df.iloc[0][col] == -1]
        safe_feats = [col for col in clean_df.columns if clean_df.iloc[0][col] == 1]

        return {
            'is_phishing': is_phishing,
            'prediction_label': pred_label,
            'confidence': confidence,
            'phishing_probability': prob_phishing,
            'legitimate_probability': 1.0 - prob_phishing,
            'risk_level': risk,
            'risk_color': color,
            'dangerous_features': dangerous_feats,
            'safe_features': safe_feats,
            'model_used': artifact['name']
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        model_filename: str = "best_model.pkl",
        threshold: float = 0.5
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Performs batch inference on dataframe.
        Appends:
            - 'predicted_class' (0 or 1)
            - 'prediction_label' ('Legitimate Website' or 'Phishing Website')
            - 'phishing_probability' (float)
            - 'threat_risk_level' (str)
        """
        is_valid, msg, clean_df = self.validate_input_dataframe(df)
        if not is_valid:
            raise ValueError(f"Batch validation error: {msg}")

        artifact = self.get_model(model_filename)
        model = artifact['model']
        use_scaled = artifact.get('use_scaled', True)

        if use_scaled:
            X_input = pd.DataFrame(self.scaler.transform(clean_df), columns=self.feature_names)
        else:
            X_input = clean_df.copy()

        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_input)[:, 1]
        else:
            probs = model.predict(X_input).astype(float)

        preds = (probs >= threshold).astype(int)
        labels = ["Phishing Website" if p == 1 else "Legitimate Website" for p in preds]

        result_df = clean_df.copy()
        result_df['predicted_class'] = preds
        result_df['prediction_label'] = labels
        result_df['phishing_probability'] = np.round(probs, 4)

        summary = {
            'total_analyzed': len(df),
            'phishing_detected': int((preds == 1).sum()),
            'legitimate_verified': int((preds == 0).sum()),
            'phishing_percentage': float((preds == 1).mean() * 100),
            'model_name': artifact['name'],
            'threshold_used': threshold
        }
        return result_df, summary

    def simulate_threshold_experiment(
        self,
        test_path: str = "data/processed/test.csv",
        model_filename: str = "best_model.pkl",
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluates test set at a custom decision threshold to simulate
        ethical AI trade-offs (False Positives vs False Negatives).
        """
        test_df = pd.read_csv(test_path)
        feature_cols = [c for c in test_df.columns if c != 'target']
        X_test_raw = test_df[feature_cols]
        y_test = test_df['target'].values

        artifact = self.get_model(model_filename)
        model = artifact['model']
        use_scaled = artifact.get('use_scaled', True)

        if use_scaled:
            X_eval = pd.DataFrame(self.scaler.transform(X_test_raw), columns=self.feature_names)
        else:
            X_eval = X_test_raw.copy()

        if hasattr(model, 'predict_proba'):
            y_probs = model.predict_proba(X_eval)[:, 1]
        else:
            y_probs = model.predict(X_eval).astype(float)

        y_pred = (y_probs >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Societal / Operational impact explanation
        impact = {
            'false_positive_impact': f"At threshold {threshold:.2f}, {fp:,} legitimate websites are falsely blocked. Users face disrupted access to safe businesses, and support tickets spike.",
            'false_negative_impact': f"At threshold {threshold:.2f}, {fn:,} phishing attacks bypass security filters. Users visiting these unblocked sites risk credential theft and malware.",
            'recommendation': "For high-security zero-trust banking or enterprise firewalls, lower threshold (e.g. 0.35-0.40) to maximize recall; for consumer search engines, balance threshold (~0.50) to prevent user frustration."
        }

        return {
            'threshold': threshold,
            'model_name': artifact['name'],
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'total_samples': len(y_test),
            'impact_analysis': impact
        }


if __name__ == "__main__":
    predictor = PhishGuardPredictor()
    # Test sample prediction
    sample_df = pd.read_csv("data/sample_phishing.csv")
    single_res = predictor.predict_single(sample_df.iloc[0].to_dict())
    print("Single Sample Test Prediction:")
    print(single_res)
    # Test batch prediction
    batch_df = pd.read_csv("data/sample_batch_test.csv")
    b_res, b_sum = predictor.predict_batch(batch_df)
    print("Batch Prediction Test Summary:")
    print(b_sum)
    # Test threshold experiment
    sim_res = predictor.simulate_threshold_experiment(threshold=0.3)
    print("Threshold Simulation (T=0.30):")
    print(f"Recall: {sim_res['recall']:.4f}, Precision: {sim_res['precision']:.4f}, FP: {sim_res['fp']}, FN: {sim_res['fn']}")
