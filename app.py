import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import shap
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="T2DM Clinical Prediction | MIDE 2026",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size:1.8rem; font-weight:700; color:#1a3c5e;
        border-bottom:3px solid #2196F3; padding-bottom:0.4rem; }
    .risk-high { background:#fff0f0; border-radius:10px;
        padding:1rem; border-left:4px solid #e53935; }
    .risk-low  { background:#f0fff4; border-radius:10px;
        padding:1rem; border-left:4px solid #43a047; }
    footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load Model (native LightGBM format — version-independent) ────────────────
@st.cache_resource(show_spinner="⏳ Loading clinical model...")
def load_model():
    base = os.path.dirname(__file__)
    booster = lgb.Booster(model_file=os.path.join(base, "model_lgbm.txt"))
    with open(os.path.join(base, "feature_names.json")) as f:
        feature_names = json.load(f)
    explainer = shap.TreeExplainer(booster)
    return booster, explainer, feature_names

try:
    booster, explainer, feature_names = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="main-header">🩺 T2DM Clinical Prediction &amp; SHAP Explanation Dashboard</p>',
    unsafe_allow_html=True)
st.markdown(
    "An **Explainable AI** clinical decision support tool — "
    "*An XAI-Driven Universal Framework for Diabetes Prediction* (MIDE 2026).")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Patient Clinical Profile")
    age    = st.slider("Age (years)", 1, 100, 52)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    bmi    = st.slider("BMI", 10.0, 60.0, 27.5, step=0.1)
    st.markdown("---")
    st.subheader("Medical History")
    hypertension  = st.selectbox("Hypertension",  [0, 1], format_func=lambda x: "Yes" if x else "No")
    heart_disease = st.selectbox("Heart Disease",  [0, 1], format_func=lambda x: "Yes" if x else "No")
    smoking       = st.selectbox("Smoking History",
                                 ["never", "former", "current", "not current", "ever", "No Info"])
    st.markdown("---")
    st.subheader("🔬 Lab Results  *(Critical)*")
    hba1c   = st.slider("HbA1c Level (%)",       3.0, 12.0, 5.8, step=0.1)
    glucose = st.slider("Blood Glucose (mg/dL)", 50,  350,  140)

gender_map  = {"Female": 0, "Male": 1, "Other": 2}
smoking_map = {"No Info": -1, "never": 0, "not current": 1,
               "former": 2, "ever": 3, "current": 4}

input_data = {
    "age": age, "hypertension": hypertension, "heart_disease": heart_disease,
    "bmi": bmi, "HbA1c_level": hba1c, "blood_glucose_level": glucose,
    "gender_enc": gender_map[gender], "smoking_enc": smoking_map[smoking],
}
input_df = pd.DataFrame([input_data])[feature_names]

# ── Prediction ────────────────────────────────────────────────────────────────
prob = float(booster.predict(input_df.values)[0])
prediction = int(prob > 0.5)

col_pred, col_shap = st.columns([1, 2], gap="large")

with col_pred:
    st.subheader("📊 Risk Assessment")
    if prediction == 1:
        st.markdown(
            f'<div class="risk-high"><h3>⚠️ High Diabetes Risk</h3>'
            f'<p style="font-size:2rem;font-weight:700;color:#e53935;">{prob*100:.1f}%</p>'
            f'<p>Probability of T2DM</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="risk-low"><h3>✅ Low Diabetes Risk</h3>'
            f'<p style="font-size:2rem;font-weight:700;color:#43a047;">{prob*100:.1f}%</p>'
            f'<p>Probability of T2DM</p></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**📌 ADA Clinical Thresholds:**")
    st.markdown("""
- 🔴 **HbA1c ≥ 6.5%** → Diabetes  
- 🟡 **HbA1c 5.7–6.4%** → Prediabetes  
- 🔴 **Glucose ≥ 126 mg/dL** → Diabetes (fasting)
    """)

with col_shap:
    st.subheader("🧠 SHAP Clinical Explanation")
    st.caption("Why did the model make this prediction? Feature-level breakdown for this patient.")

    sv = explainer.shap_values(input_df.values)
    if isinstance(sv, list):
        sv = sv[1]
    bv = explainer.expected_value
    if isinstance(bv, (list, np.ndarray)):
        bv = float(bv[1] if len(bv) > 1 else bv[0])
    else:
        bv = float(bv)

    shap_obj = shap.Explanation(
        values=sv[0], base_values=bv,
        data=input_df.values[0], feature_names=feature_names)

    fig, _ = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(shap_obj, show=False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.divider()
st.markdown(
    "📄 *MIDE 2026 — **An XAI-Driven Universal Framework for Diabetes Prediction** "
    "| LightGBM + Borderline-SMOTE + SHAP*")
