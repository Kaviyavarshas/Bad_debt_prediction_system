import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from catboost import CatBoostClassifier

# =====================================================================
# 1. PAGE CONFIGURATION & SINGLE-PAGE COMPACT THEME
# =====================================================================
st.set_page_config(
    page_title="Bad Debt Inference System",
    page_icon="🎯",
    layout="centered" # Keeps everything packed tightly into a unified presentation column
)

# Deep CSS compaction to remove whitespace and prevent vertical scrolling
st.markdown("""
    <style>
    /* Strip default Streamlit header and layout padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0rem !important;
        max-width: 850px !important;
    }
    #MainMenu, header, footer {
        visibility: hidden;
        height: 0;
    }
    
    /* Title Compacting */
    .main-title {
        font-size: 28px !important;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 2px !important;
    }
    .sub-instruction {
        font-size: 13px !important;
        color: #475569;
        margin-bottom: 6px !important;
    }
    .divider-line {
        border-top: 1px solid #e2e8f0;
        margin-bottom: 10px !important;
    }
    .section-header {
        font-size: 16px !important;
        font-weight: 700;
        color: #1e293b;
        margin-top: 8px !important;
        margin-bottom: 6px !important;
    }
    
    /* Fused Input Box styling */
    div[data-testid="stTextInput"] {
        margin-bottom: 10px !important;
    }
    
    /* Found Alert Box */
    .record-alert {
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
        padding: 6px 14px !important;
        border-radius: 4px;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 10px !important;
    }
    
    /* Compressed Scorecard elements */
    .status-box {
        background-color: #f0fdf4;
        border-radius: 6px;
        padding: 12px 16px !important;
    }
    .status-box.risk-high {
        background-color: #fef2f2;
    }
    .metric-title {
        font-size: 12px !important;
        font-weight: 700;
        color: #475569;
        margin: 0 0 4px 0 !important;
    }
    .metric-value-text {
        font-size: 18px !important;
        font-weight: 800;
        color: #0f172a;
        margin: 0 !important;
    }
    .confidence-box {
        padding-left: 5px;
    }
    .confidence-value {
        font-size: 28px !important;
        font-weight: 800;
        color: #0f172a;
        line-height: 1;
        margin-bottom: 4px !important;
    }
    .raw-risk-tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background-color: #f0fdf4;
        color: #166534;
        padding: 2px 8px !important;
        border-radius: 4px;
        font-size: 11px !important;
        font-weight: 700;
    }
    .raw-risk-tag.risk-high {
        background-color: #fef2f2;
        color: #991b1b;
    }
    
    /* Standardize space above the progress bar */
    .stProgress {
        margin-top: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. DATA & CHAMPION ARTIFACT LOADING LIFECYCLE
# =====================================================================
@st.cache_resource
def load_champion_artifacts():
    model_path = "best_catboost.cbm"
    features_path = "model_features.pkl"
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        return None, None
    model = CatBoostClassifier()
    model.load_model(model_path)
    features_blueprint = joblib.load(features_path)
    return model, features_blueprint

@st.cache_data
def load_production_database():
    data_path = "merged_production_data.parquet"
    if not os.path.exists(data_path):
        return None
    return pd.read_parquet(data_path)

model, feature_columns = load_champion_artifacts()
df_prod = load_production_database()

if model is None or df_prod is None:
    st.error("❌ Core Pipeline Deployment Artifacts Missing in Workspace Directory!")
    st.stop()

# =====================================================================
# 3. INTERFACE HEADER & SEARCH INPUT LAYERING
# =====================================================================
st.markdown('<div class="main-title">🎯 BAD DEBT INFERENCE SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-instruction">Enter a Customer ID to retrieve features and predict risk.</div>', unsafe_allow_html=True)
st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-header">🔍 Search Customer Entity ID:</div>', unsafe_allow_html=True)

search_input = st.text_input(
    label="Search Field", 
    value="", 
    placeholder="Give the customer ID here...", 
    label_visibility="collapsed"
).strip()

if not search_input:
    st.stop()

# =====================================================================
# 4. CONDITIONAL RENDER PIPELINE (TRIGGERS POST-INPUT ONLY)
# =====================================================================
matched_rows = df_prod[df_prod['Customer'].astype(str) == search_input]

if matched_rows.empty:
    st.markdown(f'<div class="record-alert" style="background-color: #fef2f2; color: #991b1b; border-color: #fca5a5;">❌ Record Not Found for Entity ID: {search_input}</div>', unsafe_allow_html=True)
    st.stop()
else:
    st.markdown('<div class="record-alert">✅ Record Found!</div>', unsafe_allow_html=True)

customer_record = matched_rows.iloc[0].copy()

# Horizontal Individual Record Table Display
st.markdown('<div class="section-header">📁 Full Merged Customer Record:</div>', unsafe_allow_html=True)

# Build a compact preview matrix
preview_df = pd.DataFrame([customer_record]).set_index('Customer', drop=False)
preview_df = preview_df.astype(str)
preview_df = preview_df.replace(['None', 'none', 'nan', 'NaN', 'NULL', 'null', '', '<NA>', 'nat', 'NaT'], 'NaN')

# Hard row containment limit (75px) to prevent layout shifting or grid leakage
st.dataframe(preview_df, use_container_width=True, height=75)

# Risk Scorecard Analysis Engine
st.markdown('<div class="section-header">📊 System Scorecard Analysis</div>', unsafe_allow_html=True)

try:
    inference_df = pd.DataFrame([customer_record]).reindex(columns=feature_columns)
    cat_cols = inference_df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in cat_cols:
        inference_df[col] = inference_df[col].astype('category').cat.codes
        
    raw_risk_prob = model.predict_proba(inference_df)[0][1]
    prediction_class = model.predict(inference_df)[0]
    
except Exception as e:
    st.error(f"❌ Structural feature template alignment failed: {str(e)}")
    st.stop()

# Determine dynamic layout parameters based on risk engine findings
if prediction_class == 1:
    status_text = "BAD DEBT RISK"
    box_class = "status-box risk-high"
    tag_class = "raw-risk-tag risk-high"
    arrow = "↑"
    display_confidence = raw_risk_prob * 100
else:
    status_text = "GOOD PROFILE"
    box_class = "status-box"
    tag_class = "raw-risk-tag"
    arrow = "↑"
    display_confidence = (1 - raw_risk_prob) * 100

# Generate Side-by-Side Visual Grid Cards
col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.markdown(f"""
        <div class="{box_class}">
            <p class="metric-title">Final Risk Status:</p>
            <p class="metric-value-text">{status_text}</p>
        </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown(f"""
        <div class="confidence-box">
            <p style="font-size: 12px; font-weight: 600; color: #475569; margin: 0 0 2px 0;">Confidence Score</p>
            <div class="confidence-value">{display_confidence:.2f}%</div>
            <div class="{tag_class}">{arrow} Raw Risk: {raw_risk_prob:.4f}</div>
        </div>
    """, unsafe_allow_html=True)

# Smooth progress meter line capping off the dashboard panel base
st.progress(float(raw_risk_prob))