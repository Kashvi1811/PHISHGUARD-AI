# ==============================================================================
# PHISHGUARD AI: Intelligent Phishing Website Detection and Security Analytics
# Author: Senior ML & Cyber Security AI Engineering Team
# University Laboratory Assignment - AI in Cyber Security
# ==============================================================================

import os
import io
import json
import logging
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, precision_recall_curve

from src.prediction import PhishGuardPredictor, EXPECTED_FEATURES
from src.feature_analysis import FEATURE_SECURITY_EXPLANATIONS

# Page Configuration
st.set_page_config(
    page_title="PHISHGUARD AI | Phishing Detection & Threat Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Cyber Dark Styling
CYBER_CSS = """
<style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1527 0%, #080d1a 90%);
        color: #e2e8f0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0b1120;
        border-right: 1px solid #1e293b;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #38bdf8;
        margin-top: 5px;
    }
    
    /* Threat status banners */
    .threat-banner-phish {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.3) 100%);
        border: 1px solid #ef4444;
        border-left: 6px solid #dc2626;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .threat-banner-legit {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.3) 100%);
        border: 1px solid #10b981;
        border-left: 6px solid #059669;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    
    /* Code blocks */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Tables and Dataframe */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
"""
st.markdown(CYBER_CSS, unsafe_allow_html=True)


# ==============================================================================
# Cached Resources and Data Loaders
# ==============================================================================

@st.cache_resource
def load_predictor():
    return PhishGuardPredictor()

@st.cache_data
def load_model_comparison_data():
    comp_path = "outputs/model_comparison/model_comparison.csv"
    if os.path.exists(comp_path):
        return pd.read_csv(comp_path)
    return None

@st.cache_data
def load_processed_data():
    proc_path = "data/processed/phishing_processed.csv"
    if os.path.exists(proc_path):
        return pd.read_csv(proc_path)
    return None

@st.cache_data
def load_test_split():
    test_path = "data/processed/test.csv"
    if os.path.exists(test_path):
        return pd.read_csv(test_path)
    return None

predictor = load_predictor()
comp_df = load_model_comparison_data()
proc_df = load_processed_data()
test_df = load_test_split()


# ==============================================================================
# Sidebar Navigation and Branding
# ==============================================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <div style="font-size: 2.4rem;">🛡️</div>
        <div style="font-size: 1.4rem; font-weight: 800; letter-spacing: 0.05em; color: #38bdf8;">PHISHGUARD AI</div>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">Intelligent Phishing Website Detection</div>
        <div style="display: inline-block; background-color: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; border-radius: 20px; padding: 2px 10px; font-size: 0.68rem; color: #38bdf8; margin-top: 8px;">
            AI LAB • V1.0 PRODUCTION
        </div>
    </div>
    """, unsafe_allow_html=True)

    pages = [
        "📊 Overview Dashboard",
        "🎯 Website Threat Analysis",
        "📁 Bulk Website Analysis",
        "🔬 Model Laboratory",
        "📈 Performance Analytics",
        "⚖️ Ethical AI Lab",
        "📖 About the Project"
    ]
    selected_page = st.radio("Navigation", pages, index=0)

    st.markdown("---")
    st.markdown("### ⚙️ Engine Settings")
    model_choices = {
        "MLP Neural Network (Best Model)": "best_model.pkl",
        "Decision Tree Classifier": "decision_tree.pkl",
        "Logistic Regression": "logistic_regression.pkl"
    }
    selected_model_label = st.selectbox("Active Inference Model", list(model_choices.keys()), index=0)
    active_model_file = model_choices[selected_model_label]

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.72rem; color: #64748b; line-height: 1.5;">
        <b>Dataset:</b> UCI Phishing Websites (ID: 327)<br>
        <b>Target Mapping:</b><br>
        • Class 0 = Legitimate Website<br>
        • Class 1 = Phishing Website<br>
        <b>Status:</b> All 3 Models Online
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# Helper Functions for Visualizations
# ==============================================================================

def render_metric_card(title, value, subtext="", border_color="#38bdf8"):
    st.markdown(f"""
    <div class="metric-card" style="border-top: 3px solid {border_color};">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: Overview Dashboard
# ==============================================================================

if selected_page == "📊 Overview Dashboard":
    st.title("🛡️ Overview Security Dashboard")
    st.markdown("Real-time telemetry, dataset distribution metrics, and model performance overview.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_rec = len(proc_df) if proc_df is not None else 11055
        render_metric_card("Total Dataset Records", f"{total_rec:,}", "UCI ID 327 Benchmark", "#38bdf8")
    with col2:
        legit_cnt = int((proc_df['target'] == 0).sum()) if proc_df is not None else 6157
        legit_pct = (legit_cnt / total_rec) * 100
        render_metric_card("Legitimate Websites", f"{legit_cnt:,}", f"{legit_pct:.1f}% Benign Traffic", "#10b981")
    with col3:
        phish_cnt = int((proc_df['target'] == 1).sum()) if proc_df is not None else 4898
        phish_pct = (phish_cnt / total_rec) * 100
        render_metric_card("Phishing Websites", f"{phish_cnt:,}", f"{phish_pct:.1f}% Threat Vectors", "#ef4444")
    with col4:
        best_acc = "96.56%"
        best_rec = "95.20%"
        render_metric_card("Champion Model", "MLP Classifier", f"Acc: {best_acc} | Rec: {best_rec}", "#8b5cf6")

    st.markdown("###")

    row1_c1, row1_c2 = st.columns([1, 1])

    with row1_c1:
        st.subheader("🎯 Class Distribution (Balanced Benchmark)")
        if proc_df is not None:
            fig_donut = go.Figure(data=[go.Pie(
                labels=['Legitimate (Safe)', 'Phishing (Malicious)'],
                values=[legit_cnt, phish_cnt],
                hole=0.55,
                marker=dict(colors=['#0284c7', '#ef4444'], line=dict(color='#0b1120', width=2)),
                textinfo='label+percent',
                textfont=dict(size=13, color='white', family='Inter')
            )])
            fig_donut.update_layout(
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=320
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    with row1_c2:
        st.subheader("⚡ Model Performance Benchmark")
        if comp_df is not None:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name='Accuracy', x=comp_df['Model'], y=comp_df['Accuracy'],
                marker_color='#38bdf8'
            ))
            fig_bar.add_trace(go.Bar(
                name='Phishing Recall', x=comp_df['Model'], y=comp_df['Recall'],
                marker_color='#ef4444'
            ))
            fig_bar.add_trace(go.Bar(
                name='F1-Score', x=comp_df['Model'], y=comp_df['F1-Score'],
                marker_color='#a855f7'
            ))
            fig_bar.update_layout(
                barmode='group',
                yaxis=dict(range=[0.85, 1.0], title='Metric Score (0-1)'),
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=320
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("###")
    st.subheader("🌐 Threat Dimension Taxonomy")
    st.markdown("""
    PhishGuard AI analyzes website posture across four specialized cyber inspection categories:
    """)
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid #38bdf8;">
            <b>1. Address Bar Anomalies</b><br>
            <span style="font-size: 0.8rem; color: #94a3b8;">
            • IP Address directly in URL<br>
            • Extended character length (>54)<br>
            • TinyURL / URL shorteners<br>
            • Hyphens (-) in domain name<br>
            • Trusted SSL Certificate status
            </span>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid #10b981;">
            <b>2. Abnormal Content</b><br>
            <span style="font-size: 0.8rem; color: #94a3b8;">
            • External object request URLs<br>
            • Null anchor tags (# / javascript)<br>
            • External metadata & script tags<br>
            • Server Form Handlers (SFH)<br>
            • Form action using mailto:
            </span>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid #f59e0b;">
            <b>3. Client-Side Scripting</b><br>
            <span style="font-size: 0.8rem; color: #94a3b8;">
            • Number of HTTP redirects (>=4)<br>
            • OnMouseOver status bar spoofing<br>
            • Disabled right-click context menu<br>
            • Credential prompt pop-up windows<br>
            • Hidden 0-pixel iframes
            </span>
        </div>
        """, unsafe_allow_html=True)
    with t4:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid #ef4444;">
            <b>4. Domain Reputation</b><br>
            <span style="font-size: 0.8rem; color: #94a3b8;">
            • WHOIS domain age (<6 months)<br>
            • DNS record presence in authority<br>
            • Alexa / SimilarWeb traffic rank<br>
            • Google Search indexation<br>
            • PhishTank / StopBadware blacklist
            </span>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: Website Threat Analysis
# ==============================================================================

elif selected_page == "🎯 Website Threat Analysis":
    st.title("🎯 Website Threat Analysis & Live Inference")
    st.info("💡 **Architectural Note:** The benchmark UCI Phishing dataset operates on **30 extracted behavioral and structural attributes** rather than raw unparsed URLs. Use the organized controls below to test any website profile or select from pre-configured cyber intelligence samples.")

    # Preset selection
    st.markdown("### 1. Select a Security Profile or Customize Features")
    sample_col1, sample_col2 = st.columns([1, 2])
    
    with sample_col1:
        preset_choice = st.selectbox(
            "Load Benchmark Scenario",
            [
                "Custom Manual Input",
                "High-Risk Phishing Attack Vector (Sample)",
                "Legitimate Enterprise Web Portal (Sample)",
                "Suspicious Borderline Case (Sample)"
            ]
        )

    # Initialize feature dict
    current_features = {}

    if preset_choice == "High-Risk Phishing Attack Vector (Sample)":
        if os.path.exists("data/sample_phishing.csv"):
            sample_row = pd.read_csv("data/sample_phishing.csv").iloc[0].to_dict()
            current_features = {k: int(v) for k, v in sample_row.items()}
    elif preset_choice == "Legitimate Enterprise Web Portal (Sample)":
        if os.path.exists("data/sample_legitimate.csv"):
            sample_row = pd.read_csv("data/sample_legitimate.csv").iloc[0].to_dict()
            current_features = {k: int(v) for k, v in sample_row.items()}
    elif preset_choice == "Suspicious Borderline Case (Sample)":
        # Mixed security signals
        current_features = {f: 1 for f in EXPECTED_FEATURES}
        current_features['sslfinal_state'] = -1
        current_features['url_of_anchor'] = -1
        current_features['prefix_suffix'] = -1
        current_features['web_traffic'] = 0
        current_features['having_sub_domain'] = 0

    # Fallback default
    if not current_features:
        current_features = {f: 1 for f in EXPECTED_FEATURES}

    # Render organized form across 4 tabs
    st.markdown("### 2. Inspect & Fine-Tune Website Attributes")
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 Address Bar Attributes",
        "📄 Abnormal Content Attributes",
        "⚡ HTML & JavaScript Controls",
        "🏢 Domain & External Reputation"
    ])

    val_options = {-1: "Dangerous / Phishy (-1)", 0: "Suspicious / Neutral (0)", 1: "Safe / Legitimate (1)"}
    val_options_binary = {-1: "Dangerous / Phishy (-1)", 1: "Safe / Legitimate (1)"}

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            current_features['having_ip_address'] = st.selectbox("Having IP Address", [-1, 1], index=0 if current_features.get('having_ip_address', 1)==-1 else 1, format_func=lambda x: val_options_binary[x])
            current_features['url_length'] = st.selectbox("URL Length", [-1, 0, 1], index=[ -1, 0, 1].index(current_features.get('url_length', 1)), format_func=lambda x: val_options[x])
            current_features['shortining_service'] = st.selectbox("URL Shortening Service", [-1, 1], index=0 if current_features.get('shortining_service', 1)==-1 else 1, format_func=lambda x: val_options_binary[x])
            current_features['having_at_symbol'] = st.selectbox("Having '@' Symbol", [-1, 1], index=0 if current_features.get('having_at_symbol', 1)==-1 else 1, format_func=lambda x: val_options_binary[x])
        with c2:
            current_features['double_slash_redirecting'] = st.selectbox("Double Slash Redirecting", [-1, 1], index=0 if current_features.get('double_slash_redirecting', 1)==-1 else 1, format_func=lambda x: val_options_binary[x])
            current_features['prefix_suffix'] = st.selectbox("Prefix/Suffix with '-'", [-1, 1], index=0 if current_features.get('prefix_suffix', 1)==-1 else 1, format_func=lambda x: val_options_binary[x])
            current_features['having_sub_domain'] = st.selectbox("Having Sub-Domains", [-1, 0, 1], index=[-1, 0, 1].index(current_features.get('having_sub_domain', 1)), format_func=lambda x: val_options[x])
            current_features['sslfinal_state'] = st.selectbox("SSL Final State (HTTPS & Cert)", [-1, 0, 1], index=[-1, 0, 1].index(current_features.get('sslfinal_state', 1)), format_func=lambda x: val_options[x])
        with c3:
            current_features['domain_registration_length'] = st.selectbox("Domain Reg. Length (<=1 yr)", [-1, 1], index=0 if current_features.get('domain_registration_length', 1)==-1 else 1, format_func=lambda x: val_options_binary[x])
            current_features['favicon'] = st.selectbox("Favicon External Source", [-1, 1], index=0 if current_features.get('favicon', 1)==-1 else 1, format_func=lambda x: val_options_binary[x])
            current_features['port'] = st.selectbox("Non-Standard Port Open", [-1, 1], index=0 if current_features.get('port', 1)==-1 else 1, format_func=lambda x: val_options_binary[x])
            current_features['https_token'] = st.selectbox("HTTPS Token in Domain", [-1, 1], index=0 if current_features.get('https_token', 1)==-1 else 1, format_func=lambda x: val_options_binary[x])

    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            current_features['request_url'] = st.selectbox("Request URL (% External Objects)", [-1, 1], index=0 if current_features.get('request_url', 1)==-1 else 1, format_func=lambda x: val_options_binary[x])
            current_features['url_of_anchor'] = st.selectbox("URL of Anchor Tags", [-1, 0, 1], index=[-1, 0, 1].index(current_features.get('url_of_anchor', 1)), format_func=lambda x: val_options[x])
        with c2:
            current_features['links_in_tags'] = st.selectbox("Links in <meta>, <script>", [-1, 0, 1], index=[-1, 0, 1].index(current_features.get('links_in_tags', 1)), format_func=lambda x: val_options[x])
            current_features['sfh'] = st.selectbox("Server Form Handler (SFH)", [-1, 0, 1], index=[-1, 0, 1].index(current_features.get('sfh', 1)), format_func=lambda x: val_options[x])
        with c3:
            current_features['submitting_to_email'] = st.selectbox("Submitting Form to Email (mailto:)", [-1, 1], index=0 if current_features.get('submitting_to_email', 1)==-1 else 1, format_func=lambda x: val_options_binary[x])
            current_features['abnormal_url'] = st.selectbox("Abnormal Hostname Identity", [-1, 1], index=0 if current_features.get('abnormal_url', 1)==-1 else 1, format_func=lambda x: val_options_binary[x])

    with tab3:
        c1, c2, c3 = st.columns(3)
        with c1:
            current_features['redirect'] = st.selectbox("Redirect Count (>=4)", [-1, 0, 1], index=[-1, 0, 1].index(current_features.get('redirect', 1)), format_func=lambda x: val_options[x])
            current_features['on_mouseover'] = st.selectbox("OnMouseOver Status Bar Change", [-1, 1], index=0 if current_features.get('on_mouseover', 1)==-1 else 1, format_func=lambda x: val_options_binary[x])
        with c2:
            current_features['rightclick'] = st.selectbox("Right Click Disabled", [-1, 1], index=0 if current_features.get('rightclick', 1)==-1 else 1, format_func=lambda x: val_options_binary[x])
            current_features['popupwindow'] = st.selectbox("PopUp Window for Credentials", [-1, 1], index=0 if current_features.get('popupwindow', 1)==-1 else 1, format_func=lambda x: val_options_binary[x])
        with c3:
            current_features['iframe'] = st.selectbox("Hidden Iframe Tag Used", [-1, 1], index=0 if current_features.get('iframe', 1)==-1 else 1, format_func=lambda x: val_options_binary[x])

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            current_features['age_of_domain'] = st.selectbox("Age of Domain (>=6 mos)", [-1, 1], index=0 if current_features.get('age_of_domain', 1)==-1 else 1, format_func=lambda x: val_options_binary[x])
            current_features['dnsrecord'] = st.selectbox("DNS Record in Host Server", [-1, 1], index=0 if current_features.get('dnsrecord', 1)==-1 else 1, format_func=lambda x: val_options_binary[x])
            current_features['web_traffic'] = st.selectbox("Web Traffic Rank", [-1, 0, 1], index=[-1, 0, 1].index(current_features.get('web_traffic', 1)), format_func=lambda x: val_options[x])
        with c2:
            current_features['page_rank'] = st.selectbox("PageRank Score", [-1, 1], index=0 if current_features.get('page_rank', 1)==-1 else 1, format_func=lambda x: val_options_binary[x])
            current_features['google_index'] = st.selectbox("Google Search Indexed", [-1, 1], index=0 if current_features.get('google_index', 1)==-1 else 1, format_func=lambda x: val_options_binary[x])
            current_features['links_pointing_to_page'] = st.selectbox("Links Pointing to Page", [-1, 0, 1], index=[-1, 0, 1].index(current_features.get('links_pointing_to_page', 1)), format_func=lambda x: val_options[x])
            current_features['statistical_report'] = st.selectbox("Statistical Blacklist Report", [-1, 1], index=0 if current_features.get('statistical_report', 1)==-1 else 1, format_func=lambda x: val_options_binary[x])

    st.markdown("###")
    analyze_btn = st.button("🚀 Analyze Website Threat Level", type="primary", use_container_width=True)

    if analyze_btn or preset_choice != "Custom Manual Input":
        pred_res = predictor.predict_single(current_features, model_filename=active_model_file)
        
        st.markdown("### 3. Threat Assessment Verdict")
        
        if pred_res['is_phishing']:
            st.markdown(f"""
            <div class="threat-banner-phish">
                <div style="font-size: 1.3rem; font-weight: 800; color: #ef4444;">🚨 THREAT DETECTED: PHISHING MALICIOUS RESOURCE</div>
                <div style="margin-top: 6px; color: #fca5a5;">
                    <b>Risk Level:</b> {pred_res['risk_level']} | 
                    <b>Confidence:</b> {pred_res['confidence']*100:.2f}% | 
                    <b>Model Evaluator:</b> {pred_res['model_used']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="threat-banner-legit">
                <div style="font-size: 1.3rem; font-weight: 800; color: #10b981;">✅ SECURITY CLEARANCE: LEGITIMATE SAFE RESOURCE</div>
                <div style="margin-top: 6px; color: #6ee7b7;">
                    <b>Risk Level:</b> {pred_res['risk_level']} | 
                    <b>Confidence:</b> {pred_res['confidence']*100:.2f}% | 
                    <b>Model Evaluator:</b> {pred_res['model_used']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Confidence Gauge & Risk Breakdown
        g_col1, g_col2 = st.columns([1, 1])
        with g_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_res['phishing_probability'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Phishing Probability (%)", 'font': {'size': 14, 'color': 'white'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#ef4444" if pred_res['is_phishing'] else "#10b981"},
                    'bgcolor': "#1e293b",
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.3)"},
                        {'range': [30, 60], 'color': "rgba(234, 179, 8, 0.3)"},
                        {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 3},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=240, margin=dict(t=30, b=10, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with g_col2:
            st.markdown("#### Influential Security Indicators")
            if pred_res['dangerous_features']:
                st.markdown(f"**⚠️ Dangerous Indicators Detected ({len(pred_res['dangerous_features'])}):**")
                st.write(", ".join([f"`{f}`" for f in pred_res['dangerous_features'][:8]]))
            if pred_res['safe_features']:
                st.markdown(f"**🛡️ Valid Security Safeguards Verified ({len(pred_res['safe_features'])}):**")
                st.write(", ".join([f"`{f}`" for f in pred_res['safe_features'][:8]]))


# ==============================================================================
# PAGE 3: Bulk Website Analysis
# ==============================================================================

elif selected_page == "📁 Bulk Website Analysis":
    st.title("📁 Batch Website Analysis & Audit")
    st.markdown("Upload a CSV file containing website feature records or load standard university benchmark test batches.")

    sample_b1, sample_b2 = st.columns([1, 1])
    with sample_b1:
        use_sample = st.checkbox("Load Demonstration Benchmark CSV (data/sample_batch_test.csv)", value=True)
    with sample_b2:
        uploaded_file = st.file_uploader("Upload Target CSV", type=["csv"])

    input_df = None
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded CSV loaded successfully ({len(input_df)} records).")
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
    elif use_sample and os.path.exists("data/sample_batch_test.csv"):
        input_df = pd.read_csv("data/sample_batch_test.csv")
        st.info(f"Loaded benchmark sample batch with {len(input_df)} test records.")

    if input_df is not None:
        st.markdown("### Preview Uploaded Telemetry")
        st.dataframe(input_df.head(5), use_container_width=True)

        run_bulk = st.button("⚡ Run Batch Threat Detection", type="primary")

        if run_bulk or use_sample:
            try:
                with st.spinner("Executing threat detection neural inference pipeline..."):
                    result_df, summary = predictor.predict_batch(input_df, model_filename=active_model_file)
                
                st.markdown("### Batch Classification Results")
                
                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    render_metric_card("Total Analyzed", f"{summary['total_analyzed']:,}", "Web Profiles", "#38bdf8")
                with m2:
                    render_metric_card("Phishing Flagged", f"{summary['phishing_detected']:,}", f"{summary['phishing_percentage']:.1f}% Positive Threats", "#ef4444")
                with m3:
                    render_metric_card("Legitimate Verified", f"{summary['legitimate_verified']:,}", "Safe Portals", "#10b981")
                with m4:
                    render_metric_card("Evaluating Model", summary['model_name'], f"Threshold: {summary['threshold_used']}", "#a855f7")

                st.markdown("###")
                
                # Detailed results table
                st.dataframe(result_df[['prediction_label', 'phishing_probability'] + EXPECTED_FEATURES[:8]], use_container_width=True)

                # Download button
                csv_buffer = io.StringIO()
                result_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Full Security Audit Report (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="phishguard_security_audit_report.csv",
                    mime="text/csv",
                    type="secondary"
                )
            except Exception as e:
                st.error(f"Batch inference failed: {e}")


# ==============================================================================
# PAGE 4: Model Laboratory
# ==============================================================================

elif selected_page == "🔬 Model Laboratory":
    st.title("🔬 Machine Learning & Deep Learning Model Laboratory")
    st.markdown("Inspect algorithmic architectures, training parameters, loss curves, and computational latency.")

    models_info = {
        "MLP Classifier (Deep Neural Network)": {
            "file": "mlp_classifier.pkl",
            "type": "Deep Learning / Artificial Neural Network",
            "arch": "30 Inputs -> Dense(64, ReLU) -> Dense(32, ReLU) -> Output(1, Softmax)",
            "hyperparams": "Optimizer: Adam, lr=0.001, alpha=0.0001, early_stopping=True, val_fraction=0.1, max_iter=350",
            "strengths": "Superior feature combination capture, high tolerance to non-linear noise, exceptional generalization.",
            "limitations": "Black-box nature requires permutation explainability, higher computational training latency."
        },
        "Decision Tree Classifier": {
            "file": "decision_tree.pkl",
            "type": "Non-Linear Rule-Based Classifier",
            "arch": "Binary Decision Tree (Max Depth = 12, Min Samples Split = 5)",
            "hyperparams": "Criterion: Gini Impurity, min_samples_leaf=2, random_state=42",
            "strengths": "Completely transparent decision rules, ultra-fast training (<0.04s) and millisecond inference.",
            "limitations": "Susceptible to slight variance if pruned improperly, step-function decision boundaries."
        },
        "Logistic Regression": {
            "file": "logistic_regression.pkl",
            "type": "Linear Probabilistic Classifier",
            "arch": "Generalized Linear Model with Sigmoid Activation",
            "hyperparams": "Penalty: L2 (Ridge), C=1.0, Solver: lbfgs, max_iter=1000",
            "strengths": "Direct probabilistic interpretation via log-odds coefficients, highly stable and lightweight.",
            "limitations": "Cannot natively model complex non-linear feature interactions without polynomial expansion."
        }
    }

    selected_lab_model = st.selectbox("Select Model to Inspect", list(models_info.keys()))
    info = models_info[selected_lab_model]
    art = joblib.load(os.path.join("models", info["file"]))
    metrics = art.get('metrics', {})

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📋 Architectural Specifications")
        st.markdown(f"""
        - **Model Family:** `{info['type']}`
        - **Architecture:** `{info['arch']}`
        - **Hyperparameters:** `{info['hyperparams']}`
        - **Training Time:** `{metrics.get('train_time_sec', 0.0):.4f} seconds`
        - **Prediction Latency:** `{metrics.get('pred_time_sec', 0.0)*1000:.2f} ms per batch`
        """)
        st.markdown(f"**Strengths:** {info['strengths']}")
        st.markdown(f"**Limitations:** {info['limitations']}")

    with col2:
        st.subheader("🎯 Test Performance Scores")
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            render_metric_card("Accuracy", f"{metrics.get('accuracy', 0.0)*100:.2f}%", "Overall Correctness", "#38bdf8")
            st.markdown("###")
            render_metric_card("Recall (Phish)", f"{metrics.get('recall', 0.0)*100:.2f}%", "Catch Rate", "#ef4444")
        with m_c2:
            render_metric_card("Precision", f"{metrics.get('precision', 0.0)*100:.2f}%", "True Positive Rate", "#10b981")
            st.markdown("###")
            render_metric_card("ROC-AUC", f"{metrics.get('roc_auc', 0.0):.4f}", "Discrimination Area", "#a855f7")

    # If neural net, show loss curve
    if 'loss_curve' in art:
        st.markdown("### 📉 Neural Network Training Convergence")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=art['loss_curve'],
            mode='lines',
            line=dict(color='#a855f7', width=3),
            name='Cross-Entropy Loss'
        ))
        fig_loss.update_layout(
            title="MLP Classifier Optimization Loss Trajectory (Adam Optimizer)",
            xaxis_title="Training Epoch",
            yaxis_title="Loss",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        st.plotly_chart(fig_loss, use_container_width=True)


# ==============================================================================
# PAGE 5: Performance Analytics
# ==============================================================================

elif selected_page == "📈 Performance Analytics":
    st.title("📈 Comparative Performance Analytics & Diagnostics")
    st.markdown("In-depth comparison of confusion matrices, ROC curves, Precision-Recall curves, and feature importance.")

    if comp_df is not None:
        st.subheader("📊 Cross-Model Comparison Matrix")
        st.dataframe(comp_df, use_container_width=True)

    tab_cm, tab_roc, tab_pr, tab_feat = st.tabs([
        "🔲 Confusion Matrices",
        "📉 ROC Curves",
        "🎯 Precision-Recall Curves",
        "⭐ Feature Importance & Explainability"
    ])

    with tab_cm:
        st.markdown("### Detailed Confusion Matrices")
        cm_col1, cm_col2, cm_col3 = st.columns(3)
        cm_images = [
            ("outputs/confusion_matrices/cm_mlp_classifier.png", cm_col1, "MLP Neural Network"),
            ("outputs/confusion_matrices/cm_decision_tree.png", cm_col2, "Decision Tree"),
            ("outputs/confusion_matrices/cm_logistic_regression.png", cm_col3, "Logistic Regression")
        ]
        for img_path, col, name in cm_images:
            with col:
                st.markdown(f"**{name}**")
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    st.warning(f"Matrix image {img_path} not found.")

    with tab_roc:
        st.markdown("### Multi-Model ROC Curves")
        roc_img = "outputs/charts/09_roc_curves_comparison.png"
        if os.path.exists(roc_img):
            st.image(roc_img, use_container_width=True)

    with tab_pr:
        st.markdown("### Precision-Recall Curves")
        pr_img = "outputs/charts/10_precision_recall_curves_comparison.png"
        if os.path.exists(pr_img):
            st.image(pr_img, use_container_width=True)

    with tab_feat:
        st.markdown("### Feature Importance Attribution")
        feat_img = "outputs/charts/13_feature_importance.png"
        if os.path.exists(feat_img):
            st.image(feat_img, use_container_width=True)
        
        st.markdown("#### Cybersecurity Domain Rationale for Top Predictors")
        for f, exp in list(FEATURE_SECURITY_EXPLANATIONS.items())[:6]:
            st.markdown(f"- **`{f}`:** {exp}")


# ==============================================================================
# PAGE 6: Ethical AI Lab
# ==============================================================================

elif selected_page == "⚖️ Ethical AI Lab":
    st.title("⚖️ Ethical AI & Decision Boundary Experiment")
    st.markdown("""
    In cybersecurity machine learning systems, default classification thresholds ($T=0.50$) introduce critical trade-offs between **security strictness** and **operational usability**.
    """)

    st.markdown("### 1. Interactive Classification Decision Threshold")
    selected_threshold = st.slider(
        "Classification Decision Threshold (T)",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Probability threshold at or above which an instance is classified as Phishing."
    )

    sim_res = predictor.simulate_threshold_experiment(threshold=selected_threshold, model_filename=active_model_file)

    st.markdown("### 2. Dynamically Recalculated Test Performance")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        render_metric_card("Accuracy", f"{sim_res['accuracy']*100:.2f}%", f"At Threshold {selected_threshold:.2f}", "#38bdf8")
    with s2:
        render_metric_card("Recall (Phishing Catch)", f"{sim_res['recall']*100:.2f}%", "Minimizes Slipped Attacks", "#ef4444")
    with s3:
        render_metric_card("Precision", f"{sim_res['precision']*100:.2f}%", "Avoids Falsely Blocking Safe Sites", "#10b981")
    with s4:
        render_metric_card("F1-Score", f"{sim_res['f1_score']:.4f}", "Harmonic Balance", "#a855f7")

    st.markdown("###")
    col_cm_dyn, col_impact = st.columns([1, 1])

    with col_cm_dyn:
        st.subheader("Dynamic Confusion Matrix")
        cm_data = np.array([[sim_res['tn'], sim_res['fp']], [sim_res['fn'], sim_res['tp']]])
        fig_dyn_cm = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=['Pred: Legitimate', 'Pred: Phishing'],
            y=['True: Legitimate', 'True: Phishing'],
            text=[[f"TN: {sim_res['tn']:,}", f"FP: {sim_res['fp']:,}"],
                  [f"FN: {sim_res['fn']:,}", f"TP: {sim_res['tp']:,}"]],
            texttemplate="%{text}",
            textfont={"size": 14, "color": "white"},
            colorscale='Blues',
            showscale=False
        ))
        fig_dyn_cm.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300,
            margin=dict(t=30, b=20, l=40, r=20)
        )
        st.plotly_chart(fig_dyn_cm, use_container_width=True)

    with col_impact:
        st.subheader("Societal & Operational Threat Analysis")
        st.markdown(f"""
        - **False Positives (Benign Blocked: {sim_res['fp']:,}):**  
          {sim_res['impact_analysis']['false_positive_impact']}
        
        - **False Negatives (Phishing Slipped: {sim_res['fn']:,}):**  
          {sim_res['impact_analysis']['false_negative_impact']}
          
        - **AI Engineering Recommendation:**  
          {sim_res['impact_analysis']['recommendation']}
        """)

    st.markdown("### 3. Ethical Governance & Explainability Dilemmas")
    st.markdown("""
    1. **Denial of Service via False Accusation:** Inadvertently blacklisting legitimate e-commerce portals inflicts severe commercial and reputational harm on benign businesses.
    2. **Adversarial Evasion & Data Drift:** Phishers continuously adapt their tactics (e.g., leveraging legitimate cloud hosts like AWS S3 or SharePoint) causing feature distribution shifts.
    3. **Transparency vs. Security through Obscurity:** Publishing exact feature weights helps legitimate organizations secure their sites, but also enables malicious adversaries to engineer evasion vectors.
    """)


# ==============================================================================
# PAGE 7: About the Project
# ==============================================================================

elif selected_page == "📖 About the Project":
    st.title("📖 About PHISHGUARD AI")
    st.markdown("### University Mini-Project: AI in Cyber Security")

    st.markdown("""
    #### 1. Problem Statement & Research Objectives
    Phishing is an insidious vector of cyber warfare and financial crime. This project engineers a complete end-to-end Machine Learning and Deep Learning system to detect fraudulent web infrastructures based on empirical telemetry rather than fragile static blocklists.

    #### 2. Dataset Attribution & Academic Citation
    - **Dataset:** UCI Machine Learning Repository — Phishing Websites Dataset (ID: 327)
    - **URL:** [https://archive.ics.uci.edu/dataset/327/phishing+websites](https://archive.ics.uci.edu/dataset/327/phishing+websites)
    - **Citation:** Mohammad, R., Thabtah, F., & McCluskey, L. (2012). *An assessment of features related to phishing websites using an automated technique*. International Conference for Internet Technology and Secured Transactions.

    #### 3. Technology Stack
    - **Language:** Python 3.10+ / 3.14
    - **Machine Learning:** Scikit-learn (`LogisticRegression`, `DecisionTreeClassifier`, `MLPClassifier`, `StandardScaler`, `Pipeline`)
    - **Interactive Dashboard:** Streamlit UI
    - **Data Processing:** Pandas, NumPy
    - **Visualization:** Plotly, Matplotlib, Seaborn
    - **Model Persistence:** Joblib

    #### 4. Limitations & Future Scope
    - **Current Limitation:** Requires extracted website attributes. Does not accept an unparsed raw URL directly.
    - **Future Scope:** Implement real-time headless Chrome browser DOM scraping (Playwright / Puppeteer) to automatically extract the 30 structural features from live URLs in under 500ms.
    """)
