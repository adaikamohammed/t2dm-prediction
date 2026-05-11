import os
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="T2DM Clinical Prediction | MIDE 2026",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem; font-weight: 700; color: #1a3c5e;
        border-bottom: 3px solid #2196F3; padding-bottom: 0.4rem;
    }
    .metric-box {
        background: #f0f7ff; border-radius: 10px;
        padding: 1rem; border-left: 4px solid #2196F3;
    }
    .risk-high { background:#fff0f0; border-left:4px solid #e53935; border-radius:10px; padding:1rem; }
    .risk-low  { background:#f0fff4; border-left:4px solid #43a047; border-radius:10px; padding:1rem; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load Pre-trained Model ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading clinical model...")
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    explainer = shap.TreeExplainer(model)
    return model, explainer, feature_names

try:
    model, explainer, feature_names = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🩺 T2DM Clinical Prediction &amp; SHAP Explanation Dashboard</p>',
            unsafe_allow_html=True)
st.markdown(
    "An **Explainable AI** clinical decision support tool based on the research paper: "
    "*An XAI-Driven Universal Framework for Diabetes Prediction* (MIDE 2026). "
    "Adjust patient data in the sidebar to receive an instant, interpretable T2DM risk assessment.",
    unsafe_allow_html=False
)
st.divider()

# ── Sidebar: Patient Input ────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Patient Clinical Profile")

    age    = st.slider("Age (years)", 1, 100, 52)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    bmi    = st.slider("BMI", 10.0, 60.0, 27.5, step=0.1)

    st.markdown("---")
    st.subheader("Medical History")
    hypertension  = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
    heart_disease = st.selectbox("Heart Disease",  [0, 1], format_func=lambda x: "Yes" if x else "No")
    smoking       = st.selectbox("Smoking History",
                                 ["never", "former", "current", "not current", "ever", "No Info"])

    st.markdown("---")
    st.subheader("🔬 Lab Results  *(Critical)*")
    hba1c   = st.slider("HbA1c Level (%)",          3.0, 12.0, 5.8, step=0.1)
    glucose = st.slider("Blood Glucose (mg/dL)",     50,  350,  140)

gender_map  = {"Female": 0, "Male": 1, "Other": 2}
smoking_map = {"No Info": -1, "never": 0, "not current": 1,
               "former": 2, "ever": 3, "current": 4}

input_data = {
    "age":               age,
    "hypertension":      hypertension,
    "heart_disease":     heart_disease,
    "bmi":               bmi,
    "HbA1c_level":       hba1c,
    "blood_glucose_level": glucose,
    "gender_enc":        gender_map[gender],
    "smoking_enc":       smoking_map[smoking],
}
input_df = pd.DataFrame([input_data])[feature_names]

# ── Prediction ────────────────────────────────────────────────────────────────
prob       = float(model.predict_proba(input_df.values)[0, 1])
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
    st.markdown("**📌 Clinical Thresholds (ADA):**")
    st.markdown("""
- 🔴 **HbA1c ≥ 6.5%** → Diabetes
- 🟡 **HbA1c 5.7–6.4%** → Prediabetes
- 🔴 **Glucose ≥ 126 mg/dL** → Diabetes (fasting)
    """)

with col_shap:
    st.subheader("🧠 SHAP Clinical Explanation")
    st.caption("Why did the model make this prediction? — Feature-level breakdown for this patient.")

    shap_vals = explainer.shap_values(input_df)
    sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
    bv = explainer.expected_value
    bv = bv[1] if isinstance(bv, (list, np.ndarray)) else bv

    shap_obj = shap.Explanation(
        values=sv, base_values=bv,
        data=input_df.values[0], feature_names=feature_names
    )
    fig, _ = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(shap_obj, show=False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.divider()
st.markdown(
    "📄 *Research Paper: **An XAI-Driven Universal Framework for Diabetes Prediction — "
    "Leveraging Bayesian Optimization and Borderline-SMOTE** | MIDE 2026 Conference*"
)
