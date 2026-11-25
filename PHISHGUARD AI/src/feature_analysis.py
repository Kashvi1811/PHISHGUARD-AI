"""
PhishGuard AI - Feature Importance and Explainability Module
Analyzes Decision Tree feature importances, Logistic Regression coefficients,
and MLP Classifier permutation importance on the test dataset.
"""

import os
import json
import logging
import joblib
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

FEATURE_SECURITY_EXPLANATIONS = {
    'sslfinal_state': 'Assesses whether the HTTPS certificate is issued by a trustworthy Certificate Authority (e.g. VeriSign, GoDaddy) and has valid duration. Phishing sites typically lack trusted certificates or use self-signed certs.',
    'url_of_anchor': 'Measures the proportion of <a> anchor tags in the webpage pointing to an external domain or null javascript/empty hashes (#). Phishing pages often copy legitimate layouts but strip anchor links.',
    'prefix_suffix': 'Detects presence of dash symbols (-) in the domain name. Phishers frequently insert dashes (e.g., paypal-security.com) to impersonate trusted corporate brands.',
    'web_traffic': 'Measures domain web traffic ranking via Alexa / SimilarWeb. Legitimate popular websites rank high; disposable short-lived phishing sites have near-zero ranking.',
    'having_sub_domain': 'Counts domain dot hierarchy. Legitimate domains have 1 or 2 dots, whereas phishers use deep multi-level subdomains (e.g. verify.bank.com.phish.ru) to deceive users.',
    'links_in_tags': 'Examines the ratio of internal vs external URLs inside <meta>, <script>, and <link> tags. Phishing templates often hotlink legitimate site scripts to appear authentic.',
    'request_url': 'Calculates the percentage of external objects (images, videos, multimedia) loaded from an external domain. Phishers load media directly from the real brand domain.',
    'domain_registration_length': 'Checks if domain registration expires within 1 year. Over 90% of phishing domains are purchased on 1-year disposable contracts.',
    'sfh': 'Server Form Handler inspection. If an authentication form submits credentials to a blank URL ("about:blank") or external server, it is a high-risk credential harvester.',
    'having_ip_address': 'Checks if an IP address (or hexadecimal notation) is used directly instead of a domain name in the address bar (e.g. http://192.168.1.1/login.php).'
}


def compute_feature_importance(
    models_dir: str = "models",
    test_path: str = "data/processed/test.csv",
    pipeline_path: str = "models/preprocessing_pipeline.pkl",
    output_charts_dir: str = "outputs/charts",
    output_reports_dir: str = "outputs/reports"
) -> Dict[str, pd.DataFrame]:
    """
    Computes feature importances across all 3 models:
    - Decision Tree: Native Gini importance
    - Logistic Regression: Normalized absolute coefficients
    - MLP: Permutation importance on the holdout test set
    """
    os.makedirs(output_charts_dir, exist_ok=True)
    os.makedirs(output_reports_dir, exist_ok=True)

    test_df = pd.read_csv(test_path)
    feature_cols = [c for c in test_df.columns if c != 'target']
    X_test_raw = test_df[feature_cols]
    y_test = test_df['target']

    pipe_dict = joblib.load(pipeline_path)
    scaler = pipe_dict['scaler']
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=feature_cols)

    importance_dfs = {}

    # 1. Decision Tree Native Importance
    dt_art = joblib.load(os.path.join(models_dir, "decision_tree.pkl"))
    dt_model = dt_art['model']
    dt_imp = pd.Series(dt_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    importance_dfs['Decision Tree'] = dt_imp

    # 2. Logistic Regression Coefficients
    lr_art = joblib.load(os.path.join(models_dir, "logistic_regression.pkl"))
    lr_model = lr_art['model']
    # Positive coefficient means higher value increases probability of phishing (class 1)
    lr_coef = pd.Series(lr_model.coef_[0], index=feature_cols).sort_values(ascending=False)
    importance_dfs['Logistic Regression'] = lr_coef

    # 3. MLP Permutation Importance on Test Set
    mlp_art = joblib.load(os.path.join(models_dir, "mlp_classifier.pkl"))
    mlp_model = mlp_art['model']
    logger.info("Computing MLP permutation importance on test set (n_repeats=5)...")
    perm_result = permutation_importance(
        mlp_model, X_test_scaled, y_test,
        n_repeats=5, random_state=42, scoring='f1'
    )
    mlp_imp = pd.Series(perm_result.importances_mean, index=feature_cols).sort_values(ascending=False)
    importance_dfs['MLP Classifier'] = mlp_imp

    # -------------------------------------------------------------
    # Generate Multi-Panel Feature Importance Visualization
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.5), dpi=300)
    
    # Decision Tree Bar
    top_dt = dt_imp.head(10)
    axes[0].barh(top_dt.index[::-1], top_dt.values[::-1], color='#059669', edgecolor='#1e293b')
    axes[0].set_title('Decision Tree: Top 10 Feature Importance (Gini)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Gini Importance', fontsize=10)

    # Logistic Regression Top Absolute Bar
    top_lr = lr_coef.abs().sort_values(ascending=False).head(10)
    bar_colors = ['#dc2626' if lr_coef[feat] > 0 else '#2563eb' for feat in top_lr.index[::-1]]
    axes[1].barh(top_lr.index[::-1], top_lr.values[::-1], color=bar_colors, edgecolor='#1e293b')
    axes[1].set_title('Logistic Regression: Top 10 Coefficients (|Weight|)\n(Red: Phishing risk, Blue: Legitimacy shield)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Absolute Coefficient Weight', fontsize=10)

    # MLP Permutation Importance Bar
    top_mlp = mlp_imp.head(10)
    axes[2].barh(top_mlp.index[::-1], top_mlp.values[::-1], color='#7c3aed', edgecolor='#1e293b')
    axes[2].set_title('MLP Neural Net: Permutation Importance (F1 Drop)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Mean F1 Score Decrease on Shuffle', fontsize=10)

    fig.suptitle('Feature Importance & Cyber Threat Explainability across Models', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    chart_path = os.path.join(output_charts_dir, "13_feature_importance.png")
    fig.savefig(chart_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved Feature Importance Chart to {chart_path}")

    # Save summary report
    report_lines = [
        "===============================================================",
        "PHISHGUARD AI - FEATURE IMPORTANCE & THREAT EXPLAINABILITY",
        "===============================================================\n",
        "--- DECISION TREE TOP 10 FEATURES ---",
        top_dt.to_string(),
        "\n--- LOGISTIC REGRESSION TOP 10 MAGNITUDE COEFFICIENTS ---",
        top_lr.to_string(),
        "\n--- MLP NEURAL NETWORK TOP 10 PERMUTATION IMPORTANCE ---",
        top_mlp.to_string(),
        "\n--- CYBERSECURITY DOMAIN RATIONALE FOR TOP INDICATORS ---"
    ]
    for feat, expl in FEATURE_SECURITY_EXPLANATIONS.items():
        report_lines.append(f"\n[+] {feat.upper()}:\n    {expl}")

    report_content = "\n".join(report_lines)
    rep_path = os.path.join(output_reports_dir, "feature_importance_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Saved Feature Importance Report to {rep_path}")

    # Save combined json
    summary_dict = {
        'decision_tree_top10': top_dt.to_dict(),
        'logistic_regression_top10': top_lr.to_dict(),
        'mlp_top10': top_mlp.to_dict(),
        'feature_explanations': FEATURE_SECURITY_EXPLANATIONS
    }
    with open("outputs/model_comparison/feature_importance.json", "w", encoding="utf-8") as jf:
        json.dump(summary_dict, jf, indent=4)

    return importance_dfs


if __name__ == "__main__":
    imps = compute_feature_importance()
    print("Feature importance analysis complete!")
