# PHISHGUARD AI: Intelligent Phishing Website Detection and Security Analytics System

> **A University Mini-Project for AI in Cyber Security**  
> *Developed as an End-to-End Cyber Threat Intelligence & Machine Learning Architecture*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/UCI%20Repo-ID%20327-green.svg)](https://archive.ics.uci.edu/dataset/327/phishing+websites)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 1. Executive Summary & Project Introduction

**PHISHGUARD AI** is an enterprise-grade artificial intelligence and deep learning solution designed to detect and classify fraudulent web resources in real time. Phishing continues to represent the primary entry point for advanced persistent threats (APTs), credential harvesting attacks, and enterprise ransomware deployment.

Traditional defenses—such as static blocklists, heuristic signature engines, and DNS sinkholes—fail when confronted with modern, dynamic, and disposable phishing infrastructure. PhishGuard AI addresses this vulnerability by deploying trained Machine Learning and Deep Learning architectures capable of identifying fraudulent patterns across 30 behavioral, structural, and reputation attributes.

---

## 2. Problem Statement & Threat Vector Landscape

Attackers frequently register short-lived disposable domain names, exploit free Certificate Authorities (e.g. Let's Encrypt), perform domain typosquatting (inserting hyphens, URL shorteners), tamper with browser status bars via JavaScript, and clone authentic brand interfaces while stripping legitimate anchor tags.

### Project Objectives:
1. **Automated Official Data Ingestion:** Programmatically acquire the benchmark UCI Phishing Websites Dataset (ID: 327) using `ucimlrepo` with automated offline/ARFF fallback.
2. **Standardized Cyber Threat Target Normalization:** Convert raw values into standard cyber classification syntax:
   - **`Class 0`:** Legitimate Website (Original UCI `+1`)
   - **`Class 1`:** Phishing Threat Vector (Original UCI `-1`)
3. **Rigorous Leakage-Free Preprocessing:** Apply Scikit-learn pipelines with `StandardScaler` fitted strictly on training data (80:20 stratified split).
4. **Multi-Model Benchmark:** Train and cross-evaluate:
   - **Logistic Regression** (L2 Regularized Linear Classifier)
   - **Decision Tree Classifier** (Gini-optimized Rule Classifier)
   - **Multi-Layer Perceptron (MLP)** (Deep Artificial Neural Network with Early Stopping)
5. **Feature Attribution & Threat Explainability:** Extract native Gini importance, linear coefficients, and permutation importance.
6. **Ethical AI & Decision Threshold Sensitivity:** Dynamically simulate the operational trade-offs between **False Positives** (blocking safe businesses) and **False Negatives** (letting attacks penetrate).
7. **Interactive Cybersecurity Dashboard:** Build an interactive, multi-page Streamlit application (**PHISHGUARD AI**).

---

## 3. Dataset Information & Academic Citation

- **Source:** [UCI Machine Learning Repository — Phishing Websites Dataset](https://archive.ics.uci.edu/dataset/327/phishing+websites)
- **Dataset ID:** `327`
- **Instances:** 11,055 records (6,157 Legitimate [55.7%], 4,898 Phishing [44.3%])
- **Attributes:** 30 discrete attributes across 4 security dimensions:
  1. *Address Bar Features:* `having_ip_address`, `url_length`, `shortining_service`, `having_at_symbol`, `double_slash_redirecting`, `prefix_suffix`, `having_sub_domain`, `sslfinal_state`, `domain_registration_length`, `favicon`, `port`, `https_token`.
  2. *Abnormal Content Features:* `request_url`, `url_of_anchor`, `links_in_tags`, `sfh`, `submitting_to_email`, `abnormal_url`.
  3. *Client-Side HTML/JS Features:* `redirect`, `on_mouseover`, `rightclick`, `popupwindow`, `iframe`.
  4. *Domain & External Reputation Features:* `age_of_domain`, `dnsrecord`, `web_traffic`, `page_rank`, `google_index`, `links_pointing_to_page`, `statistical_report`.

### Academic Citation:
```bibtex
@article{mohammad2012assessment,
  title={An assessment of features related to phishing websites using an automated technique},
  author={Mohammad, Rami M and Thabtah, Fadi and McCluskey, Lee},
  journal={International Conference for Internet Technology and Secured Transactions},
  year={2012}
}
```

---

## 4. Project Repository Structure

```text
PHISHGUARD AI/
│
├── app.py                                # Streamlit Interactive Cyber Dashboard
├── requirements.txt                      # Project dependency specification
├── README.md                             # Comprehensive project documentation
├── .gitignore                            # Version control exclusion rules
│
├── data/
│   ├── raw/                              # Original ingested UCI dataset files
│   │   ├── phishing_raw.csv
│   │   ├── raw_features.csv
│   │   ├── raw_targets.csv
│   │   └── metadata.json
│   ├── processed/                        # Normalized & stratified split datasets
│   │   ├── phishing_processed.csv
│   │   ├── train.csv                     # 80% Stratified Training Partition (8,844 rows)
│   │   └── test.csv                      # 20% Stratified Testing Partition (2,211 rows)
│   ├── dataset_info.md                   # Complete dataset dictionary & semantics
│   ├── sample_legitimate.csv             # 10 Legitimate demonstration web profiles
│   ├── sample_phishing.csv               # 10 Phishing demonstration attack profiles
│   └── sample_batch_test.csv             # Balanced batch upload test CSV
│
├── notebooks/
│   └── phishing_website_detection.ipynb  # Executed, 18-stage end-to-end Jupyter Notebook
│
├── src/
│   ├── data_loader.py                    # UCI dataset fetcher with fallback logic
│   ├── preprocessing.py                  # Normalization, stratified split & pipelines
│   ├── eda.py                            # Exploratory Data Analysis & chart generator
│   ├── train_models.py                   # Model training & persistence engine
│   ├── evaluate_models.py                # Comparative diagnostics & metrics engine
│   ├── feature_analysis.py               # Multi-model feature importance & explainability
│   ├── prediction.py                     # Inference engine, schema validator & threshold lab
│   └── build_notebook.py                 # Notebook generation & programmatic execution script
│
├── models/
│   ├── logistic_regression.pkl           # Trained Logistic Regression artifact
│   ├── decision_tree.pkl                 # Trained Decision Tree artifact
│   ├── mlp_classifier.pkl                # Trained MLP Neural Network artifact
│   ├── best_model.pkl                    # Champion model (MLP Classifier)
│   └── preprocessing_pipeline.pkl        # Fitted StandardScaler & schema pipeline
│
├── outputs/
│   ├── charts/                           # High-resolution (300 DPI) publication graphics
│   │   ├── 01_class_distribution_count.png
│   │   ├── 02_correlation_heatmap.png
│   │   ├── 03_feature_distribution_histograms.png
│   │   ├── 04_feature_boxplots.png
│   │   ├── 05_feature_comparison_legit_vs_phishing.png
│   │   ├── 06_class_distribution_donut.png
│   │   ├── 07_top_correlated_features.png
│   │   ├── 08_security_threat_dimension_analysis.png
│   │   ├── 09_roc_curves_comparison.png
│   │   ├── 10_precision_recall_curves_comparison.png
│   │   ├── 11_mlp_loss_curve.png
│   │   ├── 12_model_performance_comparison.png
│   │   └── 13_feature_importance.png
│   ├── confusion_matrices/               # Diagnostic heatmaps with labeled TP/TN/FP/FN
│   │   ├── cm_logistic_regression.png
│   │   ├── cm_decision_tree.png
│   │   └── cm_mlp_classifier.png
│   ├── reports/                          # Detailed text classification reports
│   │   ├── report_logistic_regression.txt
│   │   ├── report_decision_tree.txt
│   │   ├── report_mlp_classifier.txt
│   │   └── feature_importance_report.txt
│   └── model_comparison/                 # Quantitative benchmarking files
│       ├── model_comparison.csv
│       ├── model_comparison.json
│       ├── feature_importance.json
│       └── comparison_charts.png
│
└── tests/
    ├── test_data.py                      # Data validation & schema tests
    ├── test_models.py                    # Model integrity & benchmark tests
    └── test_prediction.py                # Single/batch prediction & threshold tests
```

---

## 5. Model Performance Results

All models were evaluated on the exact same holdout test set (**2,211 instances**; 1,231 Legitimate, 980 Phishing):

| Model Architecture | Accuracy | Precision | Phishing Recall | F1-Score | ROC-AUC | Train Time | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **92.81%** | 93.44% | 90.10% | 0.9174 | 0.9785 | **0.059s** | 3.1 ms |
| **Decision Tree Classifier** | **96.34%** | 96.58% | 95.10% | 0.9584 | 0.9868 | **0.037s** | 2.7 ms |
| **MLP Neural Network (Best)** | **96.56%** | **96.99%** | **95.20%** | **0.9609** | **0.9944** | 3.465s | 4.5 ms |

### Champion Model Selection Rationale:
In mission-critical cybersecurity operations, catching malicious campaigns (**Recall**) while avoiding user disruption (**Precision**) is paramount. The **MLP Neural Network** was selected as the **Champion Model** (`best_model.pkl`) because it achieved:
- **Highest Phishing Recall (95.20%):** Slipped past only 47 out of 980 phishing campaigns.
- **Lowest False Positives (29 out of 1,231 legitimate sites):** Exceptional precision (96.99%).
- **Superior ROC-AUC (0.9944):** Near-perfect class separability across arbitrary probability thresholds.

---

## 6. Installation & Execution Guide

### Prerequisites
- Python 3.10, 3.11, or 3.14
- Git (optional)

### 1. Clone or Open Project Directory
```powershell
cd "c:\Users\Kashvi Soni\Downloads\PHISHGUARD AI"
```

### 2. Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run Automated Test Suite
```powershell
python -m pytest tests -v
```

### 4. Execute Full Pipeline (Optional - Already Pre-Executed)
```powershell
python src/data_loader.py
python src/preprocessing.py
python src/eda.py
python src/train_models.py
python src/evaluate_models.py
python src/feature_analysis.py
python src/build_notebook.py
```

### 5. Launch the Streamlit Web Application
Run via Python module (recommended on Windows):
```powershell
python -m streamlit run app.py
```
Or run the provided launcher script:
```powershell
.\run_app.ps1
```
*(Alternatively, double-click `run_app.bat` in File Explorer).*

If you prefer using the shorthand `streamlit run app.py`, add Python Scripts to your PATH:
```powershell
$env:PATH += ";$env:APPDATA\Python\Python314\Scripts"
streamlit run app.py
```
The application will launch automatically in your browser at `http://localhost:8501`.

---

## 7. Streamlit Dashboard Features (7 Specialized Pages)

1. **📊 Overview Dashboard:** Executive telemetry, total records, class breakdown, champion model badge, donut charts, and the 4 threat dimensions.
2. **🎯 Website Threat Analysis:** Interactive form across 4 tabbed categories allowing users to test pre-configured threat scenarios or custom attributes. Displays live risk level (e.g. `CRITICAL THREAT`, `VERIFIED SAFE`), probability meter gauge, and contributing feature breakdowns.
3. **📁 Bulk Website Analysis:** Drag-and-drop batch CSV file upload with automated schema verification, batch threat detection, summary cards, and downloadable CSV audit report.
4. **🔬 Model Laboratory:** In-depth inspection of Logistic Regression, Decision Tree, and MLP architectures, hyperparameter configs, loss curves, and active model toggling.
5. **📈 Performance Analytics:** Interactive confusion matrix comparisons, multi-model ROC & Precision-Recall curves, and feature importance rankings.
6. **⚖️ Ethical AI Lab:** Interactive decision threshold slider ($T \in [0.05, 0.95]$) demonstrating dynamic re-calculation of Confusion Matrices, False Positives vs. False Negatives, and sociological governance considerations.
7. **📖 About the Project:** Problem statement, academic citations, tech stack, and future research roadmap.

---

## 8. Ethical Considerations & AI Governance

- **Denial of Service via False Accusations:** Incorrectly blocking legitimate e-commerce portals inflicts severe commercial and reputational harm on small businesses.
- **The Threshold Trade-Off:** 
  - Lowering $T \to 0.35$ minimizes dangerous False Negatives (recommended for military, banking, and enterprise zero-trust perimeters).
  - Raising $T \to 0.50 - 0.55$ prevents user friction (recommended for consumer web browsers).
- **Adversarial Drift:** Cybercriminals constantly adapt URL formats to evade detection rules; models must undergo continuous retraining against live threat feeds.

---

## 9. Academic & Laboratory Verification
This project satisfies all requirements for university-level Artificial Intelligence in Cyber Security laboratory coursework.
