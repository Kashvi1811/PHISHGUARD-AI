"""
PhishGuard AI - Exploratory Data Analysis (EDA) Module
Performs comprehensive data exploration and generates high-resolution,
publication-quality visualizations for phishing indicator analysis.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configure matplotlib styling for modern cybersecurity dashboard aesthetic
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8


def run_eda_and_generate_charts(
    processed_data_path: str = "data/processed/phishing_processed.csv",
    output_dir: str = "outputs/charts"
) -> dict:
    """
    Performs complete EDA and outputs high-resolution charts into output_dir.
    Returns a dictionary of EDA statistical summaries.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(processed_data_path)
    logger.info(f"Loaded processed data for EDA. Shape: {df.shape}")

    feature_cols = [c for c in df.columns if c != 'target']
    X = df[feature_cols]
    y = df['target']

    summary_stats = {
        "num_records": len(df),
        "num_features": len(feature_cols),
        "legitimate_count": int((y == 0).sum()),
        "phishing_count": int((y == 1).sum()),
        "legitimate_pct": float((y == 0).mean() * 100),
        "phishing_pct": float((y == 1).mean() * 100),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_records": int(df.duplicated().sum())
    }
    logger.info(f"EDA Summary: {summary_stats}")

    # Color Palette: Teal / Cyan (#0ea5e9, #0284c7) for Legitimate, Coral / Red (#ef4444, #dc2626) for Phishing
    palette = {0: '#0284c7', 1: '#ef4444', '0': '#0284c7', '1': '#ef4444'}
    class_labels = {0: 'Legitimate Website', 1: 'Phishing Website'}

    # -------------------------------------------------------------
    # Chart 1: Legitimate vs Phishing Website Distribution (Count)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    counts = y.value_counts().sort_index()
    bars = ax.bar(['Legitimate (0)', 'Phishing (1)'], counts.values, color=['#0284c7', '#ef4444'], width=0.55, edgecolor='#1e293b')
    ax.set_title('Distribution of Website Classes in Dataset', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Number of Websites', fontsize=12)
    ax.set_xlabel('Class Label', fontsize=12)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax.annotate(f'{h:,}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "01_class_distribution_count.png")
    fig.savefig(chart1_path, dpi=300)
    plt.close(fig)
    logger.info("Saved Chart 1: Class distribution count")

    # -------------------------------------------------------------
    # Chart 2: Feature Correlation Heatmap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(18, 14), dpi=300)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, annot=False,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title('Feature Correlation Matrix Heatmap (Phishing Website Attributes)', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "02_correlation_heatmap.png")
    fig.savefig(chart2_path, dpi=300)
    plt.close(fig)
    logger.info("Saved Chart 2: Correlation heatmap")

    # -------------------------------------------------------------
    # Chart 3: Feature Distribution Histograms
    # -------------------------------------------------------------
    key_features = [
        'sslfinal_state', 'url_of_anchor', 'prefix_suffix', 'web_traffic',
        'having_sub_domain', 'request_url', 'links_in_tags', 'sfh', 'domain_registration_length'
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=300)
    axes = axes.flatten()
    for i, col in enumerate(key_features):
        ax = axes[i]
        sns.histplot(data=df, x=col, hue='target', multiple='dodge', palette=palette, shrink=0.8, discrete=True, ax=ax)
        ax.set_title(f'Feature: {col}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Attribute Value (-1, 0, 1)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles=handles, labels=['Legitimate', 'Phishing'], fontsize=8)
    fig.suptitle('Distribution of Key Security Features by Class', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "03_feature_distribution_histograms.png")
    fig.savefig(chart3_path, dpi=300)
    plt.close(fig)
    logger.info("Saved Chart 3: Feature distribution histograms")

    # -------------------------------------------------------------
    # Chart 4: Boxplots of Important Features
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300)
    axes = axes.flatten()
    top_box_features = ['sslfinal_state', 'url_of_anchor', 'prefix_suffix', 'having_sub_domain', 'web_traffic', 'request_url']
    for i, col in enumerate(top_box_features):
        ax = axes[i]
        sns.boxplot(data=df, x='target', y=col, hue='target', palette=palette, ax=ax, width=0.45, legend=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Legitimate (0)', 'Phishing (1)'])
        ax.set_title(f'Boxplot: {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Website Class', fontsize=10)
        ax.set_ylabel(f'{col} Value', fontsize=10)
    fig.suptitle('Boxplot Analysis of Critical Phishing Predictors', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    chart4_path = os.path.join(output_dir, "04_feature_boxplots.png")
    fig.savefig(chart4_path, dpi=300)
    plt.close(fig)
    logger.info("Saved Chart 4: Feature boxplots")

    # -------------------------------------------------------------
    # Chart 5: Feature-wise Comparison (Mean values Legit vs Phishing)
    # -------------------------------------------------------------
    mean_by_class = df.groupby('target')[feature_cols].mean().T
    mean_by_class.columns = ['Legitimate', 'Phishing']
    mean_by_class['diff'] = (mean_by_class['Legitimate'] - mean_by_class['Phishing']).abs()
    top_diff_features = mean_by_class.sort_values(by='diff', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    x_idx = np.arange(len(top_diff_features))
    w = 0.38
    ax.bar(x_idx - w/2, top_diff_features['Legitimate'], width=w, label='Legitimate (0)', color='#0284c7', edgecolor='#1e293b')
    ax.bar(x_idx + w/2, top_diff_features['Phishing'], width=w, label='Phishing (1)', color='#ef4444', edgecolor='#1e293b')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(top_diff_features.index, rotation=45, ha='right', fontsize=10)
    ax.set_title('Top 15 Feature Attribute Comparison: Legitimate vs Phishing Websites', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Mean Encoded Attribute Value', fontsize=11)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.legend(fontsize=11)
    plt.tight_layout()
    chart5_path = os.path.join(output_dir, "05_feature_comparison_legit_vs_phishing.png")
    fig.savefig(chart5_path, dpi=300)
    plt.close(fig)
    logger.info("Saved Chart 5: Feature comparison legit vs phishing")

    # -------------------------------------------------------------
    # Chart 6: Class Distribution Donut Chart
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    colors = ['#0284c7', '#ef4444']
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=['Legitimate Websites', 'Phishing Websites'],
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        textprops=dict(color='#0f172a', fontsize=12, fontweight='bold'),
        wedgeprops=dict(width=0.45, edgecolor='#1e293b')
    )
    plt.setp(autotexts, size=13, weight="bold", color="white")
    ax.set_title('Dataset Target Composition\n(UCI ID 327)', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    chart6_path = os.path.join(output_dir, "06_class_distribution_donut.png")
    fig.savefig(chart6_path, dpi=300)
    plt.close(fig)
    logger.info("Saved Chart 6: Class distribution donut")

    # -------------------------------------------------------------
    # Chart 7: Top Correlated Features with Phishing Target
    # -------------------------------------------------------------
    target_corr = df.corr()['target'].drop('target').sort_values()
    top_corr = pd.concat([target_corr.head(7), target_corr.tail(8)])

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    bar_colors = ['#0284c7' if val < 0 else '#ef4444' for val in top_corr.values]
    bars = ax.barh(top_corr.index, top_corr.values, color=bar_colors, edgecolor='#1e293b', height=0.6)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_title('Correlation of Key Features with Phishing Target (Target = 1)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Pearson Correlation Coefficient', fontsize=11)
    for bar in bars:
        w = bar.get_width()
        ha = 'right' if w < 0 else 'left'
        offset = -0.02 if w < 0 else 0.02
        ax.annotate(f'{w:.3f}',
                    xy=(w + offset, bar.get_y() + bar.get_height() / 2),
                    va='center', ha=ha, fontsize=9, fontweight='bold')
    plt.tight_layout()
    chart7_path = os.path.join(output_dir, "07_top_correlated_features.png")
    fig.savefig(chart7_path, dpi=300)
    plt.close(fig)
    logger.info("Saved Chart 7: Top correlated features")

    # -------------------------------------------------------------
    # Chart 8: Security Threat Dimension Analysis
    # -------------------------------------------------------------
    dimensions = {
        'Address Bar': ['sslfinal_state', 'prefix_suffix', 'having_sub_domain', 'url_length'],
        'Abnormal Content': ['url_of_anchor', 'links_in_tags', 'request_url', 'sfh'],
        'HTML & JavaScript': ['redirect', 'iframe', 'popupwindow', 'on_mouseover'],
        'Domain Reputation': ['web_traffic', 'page_rank', 'google_index', 'age_of_domain']
    }
    dim_means = {}
    for dim, feats in dimensions.items():
        legit_mean = df[df['target'] == 0][feats].mean().mean()
        phish_mean = df[df['target'] == 1][feats].mean().mean()
        dim_means[dim] = {'Legitimate': legit_mean, 'Phishing': phish_mean}
    dim_df = pd.DataFrame(dim_means).T

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    x = np.arange(len(dim_df))
    w = 0.35
    ax.bar(x - w/2, dim_df['Legitimate'], width=w, label='Legitimate Websites (Safe)', color='#0284c7', edgecolor='#1e293b')
    ax.bar(x + w/2, dim_df['Phishing'], width=w, label='Phishing Websites (Malicious)', color='#ef4444', edgecolor='#1e293b')
    ax.set_xticks(x)
    ax.set_xticklabels(dim_df.index, fontsize=11, fontweight='bold')
    ax.set_title('Security Dimension Average Scores Across Website Classes', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Mean Security Score (-1: Dangerous, +1: Safe)', fontsize=11)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.legend(fontsize=10)
    plt.tight_layout()
    chart8_path = os.path.join(output_dir, "08_security_threat_dimension_analysis.png")
    fig.savefig(chart8_path, dpi=300)
    plt.close(fig)
    logger.info("Saved Chart 8: Security threat dimensions")

    return summary_stats


if __name__ == "__main__":
    stats = run_eda_and_generate_charts()
    print("EDA charts generated successfully!")
    print(stats)
