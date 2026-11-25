"""
PhishGuard AI - Model Evaluation & Comparison Module
Computes comparative performance metrics, confusion matrices (with TP/TN/FP/FN),
ROC curves, Precision-Recall curves, loss curves, and generates CSV/JSON reports.
"""

import os
import json
import logging
import joblib
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'


def generate_evaluation_artifacts(
    models_dir: str = "models",
    test_path: str = "data/processed/test.csv",
    pipeline_path: str = "models/preprocessing_pipeline.pkl",
    output_charts_dir: str = "outputs/charts",
    output_cm_dir: str = "outputs/confusion_matrices",
    output_reports_dir: str = "outputs/reports",
    output_comp_dir: str = "outputs/model_comparison"
) -> pd.DataFrame:
    """
    Evaluates all trained models on test set, generates:
    - Detailed Confusion Matrix graphics for each model
    - ROC curves comparison plot
    - Precision-Recall curves comparison plot
    - MLP loss curve plot
    - Model comparison table (CSV & JSON)
    - Comparative metric bar charts
    """
    for d in [output_charts_dir, output_cm_dir, output_reports_dir, output_comp_dir]:
        os.makedirs(d, exist_ok=True)

    test_df = pd.read_csv(test_path)
    feature_cols = [c for c in test_df.columns if c != 'target']
    X_test_raw = test_df[feature_cols]
    y_test = test_df['target']

    pipe_dict = joblib.load(pipeline_path)
    scaler = pipe_dict['scaler']
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=feature_cols)

    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'MLP Classifier': 'mlp_classifier.pkl'
    }

    comparison_data = []
    curves_data = {}

    for name, fname in model_files.items():
        fpath = os.path.join(models_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Model file not found: {fpath}")

        artifact = joblib.load(fpath)
        model = artifact['model']
        use_scaled = artifact.get('use_scaled', True)
        X_eval = X_test_scaled if use_scaled else X_test_raw

        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1] if hasattr(model, 'predict_proba') else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        comparison_data.append({
            'Model': name,
            'Accuracy': float(acc),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1-Score': float(f1),
            'ROC-AUC': float(roc_auc),
            'Training Time (s)': float(artifact.get('train_time', 0.0)),
            'Prediction Time (s)': float(artifact.get('pred_time', 0.0)),
            'True Negatives (TN)': int(tn),
            'False Positives (FP)': int(fp),
            'False Negatives (FN)': int(fn),
            'True Positives (TP)': int(tp)
        })

        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_prob)
            curves_data[name] = {
                'fpr': fpr, 'tpr': tpr, 'auc': roc_auc,
                'precision': pr_precision, 'recall': pr_recall
            }

        # -------------------------------------------------------------
        # Confusion Matrix Plot
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
        cm_labels = np.array([
            [f"True Negative (TN)\n{tn:,}\n({(tn/len(y_test))*100:.1f}%)",
             f"False Positive (FP)\n{fp:,}\n({(fp/len(y_test))*100:.1f}%)"],
            [f"False Negative (FN)\n{fn:,}\n({(fn/len(y_test))*100:.1f}%)",
             f"True Positive (TP)\n{tp:,}\n({(tp/len(y_test))*100:.1f}%)"]
        ])
        sns.heatmap(cm, annot=cm_labels, fmt="", cmap="Blues", cbar=False,
                    xticklabels=['Pred: Legitimate', 'Pred: Phishing'],
                    yticklabels=['Actual: Legitimate', 'Actual: Phishing'],
                    ax=ax, linewidths=1.5, annot_kws={"size": 11, "weight": "bold"})
        ax.set_title(f'Confusion Matrix: {name}', fontsize=13, fontweight='bold', pad=15)
        ax.set_ylabel('True Security Class', fontsize=11)
        ax.set_xlabel('Predicted Security Class', fontsize=11)
        plt.tight_layout()
        cm_safe_name = name.lower().replace(' ', '_')
        cm_save_path = os.path.join(output_cm_dir, f"cm_{cm_safe_name}.png")
        fig.savefig(cm_save_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved Confusion Matrix for {name} to {cm_save_path}")

        # -------------------------------------------------------------
        # Text Classification Report
        # -------------------------------------------------------------
        clf_rep = classification_report(
            y_test, y_pred,
            target_names=['Legitimate Website (0)', 'Phishing Website (1)'],
            digits=4
        )
        report_text = f"""===============================================================
PHISHGUARD AI - SECURITY CLASSIFICATION REPORT: {name.upper()}
===============================================================
Accuracy:             {acc:.4f}
Precision (Phishing): {prec:.4f}
Recall (Phishing):    {rec:.4f}
F1-Score:             {f1:.4f}
ROC-AUC:              {roc_auc:.4f}
Training Time:        {artifact.get('train_time', 0.0):.4f} seconds
Prediction Time:      {artifact.get('pred_time', 0.0):.4f} seconds

Confusion Matrix Details:
- True Negatives (Benign correctly verified):  {tn:,}
- False Positives (Benign incorrectly blocked): {fp:,}
- False Negatives (Phishing missed / slipped): {fn:,}
- True Positives (Phishing successfully blocked): {tp:,}

Detailed Metrics:
{clf_rep}
"""
        with open(os.path.join(output_reports_dir, f"report_{cm_safe_name}.txt"), "w", encoding="utf-8") as rf:
            rf.write(report_text)

    # -------------------------------------------------------------
    # Comparison Table Save (CSV & JSON)
    # -------------------------------------------------------------
    comp_df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(output_comp_dir, "model_comparison.csv")
    json_path = os.path.join(output_comp_dir, "model_comparison.json")
    comp_df.to_csv(csv_path, index=False)
    comp_df.to_json(json_path, orient='records', indent=4)
    logger.info(f"Saved Model Comparison table to {csv_path} and {json_path}")

    # -------------------------------------------------------------
    # Multi-Model ROC Curves Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)
    colors = {'Logistic Regression': '#2563eb', 'Decision Tree': '#059669', 'MLP Classifier': '#7c3aed'}
    for name, cdata in curves_data.items():
        ax.plot(cdata['fpr'], cdata['tpr'], label=f"{name} (AUC = {cdata['auc']:.4f})",
                color=colors.get(name, '#000000'), linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.5000)', alpha=0.7)
    ax.set_title('Receiver Operating Characteristic (ROC) Curves Comparison', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('False Positive Rate (Legitimate incorrectly flagged)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Phishing correctly caught)', fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    roc_path = os.path.join(output_charts_dir, "09_roc_curves_comparison.png")
    fig.savefig(roc_path, dpi=300)
    plt.close(fig)
    logger.info("Saved ROC Curves Comparison")

    # -------------------------------------------------------------
    # Multi-Model Precision-Recall Curves Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)
    for name, cdata in curves_data.items():
        ax.plot(cdata['recall'], cdata['precision'], label=f"{name}",
                color=colors.get(name, '#000000'), linewidth=2)
    ax.set_title('Precision-Recall Curves Comparison (Phishing Detection)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Recall (Detection Rate)', fontsize=11)
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    pr_path = os.path.join(output_charts_dir, "10_precision_recall_curves_comparison.png")
    fig.savefig(pr_path, dpi=300)
    plt.close(fig)
    logger.info("Saved Precision-Recall Curves Comparison")

    # -------------------------------------------------------------
    # MLP Loss Curve Plot
    # -------------------------------------------------------------
    mlp_art = joblib.load(os.path.join(models_dir, "mlp_classifier.pkl"))
    if 'loss_curve' in mlp_art:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.plot(mlp_art['loss_curve'], color='#7c3aed', linewidth=2, label='Training Loss')
        ax.set_title('MLP Neural Network Training Loss Curve', fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Epoch Iteration', fontsize=11)
        ax.set_ylabel('Cross-Entropy Loss', fontsize=11)
        ax.legend(fontsize=10)
        plt.tight_layout()
        loss_path = os.path.join(output_charts_dir, "11_mlp_loss_curve.png")
        fig.savefig(loss_path, dpi=300)
        plt.close(fig)
        logger.info("Saved MLP Loss Curve")

    # -------------------------------------------------------------
    # Performance Comparison Bar Charts
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), dpi=300)
    metrics_to_plot = [
        ('Accuracy', 'Accuracy Score (Higher is Better)', '#0284c7'),
        ('Precision', 'Precision Score (Higher is Better)', '#0d9488'),
        ('Recall', 'Phishing Recall Score (Catch Rate)', '#dc2626'),
        ('F1-Score', 'F1-Score (Harmonic Mean)', '#7c3aed'),
        ('ROC-AUC', 'ROC-AUC Score (Discrimination)', '#d97706'),
        ('Training Time (s)', 'Training Time in Seconds (Lower is Faster)', '#475569')
    ]
    axes = axes.flatten()
    for idx, (m_col, m_title, m_color) in enumerate(metrics_to_plot):
        ax = axes[idx]
        bars = ax.bar(comp_df['Model'], comp_df[m_col], color=m_color, width=0.55, edgecolor='#1e293b')
        ax.set_title(m_title, fontsize=11, fontweight='bold')
        ax.set_ylabel(m_col, fontsize=10)
        ax.tick_params(axis='x', rotation=15)
        for bar in bars:
            h = bar.get_height()
            fmt = f"{h:.4f}" if m_col != 'Training Time (s)' else f"{h:.3f}s"
            ax.annotate(fmt, xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        if m_col != 'Training Time (s)':
            ax.set_ylim(0, 1.12)
    fig.suptitle('Comprehensive Model Performance Comparison (PhishGuard AI)', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    comp_chart_path = os.path.join(output_comp_dir, "comparison_charts.png")
    fig.savefig(comp_chart_path, dpi=300)
    fig.savefig(os.path.join(output_charts_dir, "12_model_performance_comparison.png"), dpi=300)
    plt.close(fig)
    logger.info("Saved Performance Comparison Bar Charts")

    return comp_df


if __name__ == "__main__":
    df_res = generate_evaluation_artifacts()
    print("Model Evaluation Complete! Summary Table:")
    print(df_res.to_string())
