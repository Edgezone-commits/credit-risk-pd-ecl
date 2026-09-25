import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from src.features import add_features

st.set_page_config(page_title="Credit Default Risk & ECL Estimator", layout="centered")

# ---- Load models (cached so they don't reload on every interaction) ----
@st.cache_resource
def load_models():
    lgbm_raw = joblib.load("src/lgbm_model.joblib")
    lgbm_cal = joblib.load("src/lgbm_calibrated.joblib")
    return lgbm_raw, lgbm_cal

lgbm_raw, lgbm_cal = load_models()
explainer = shap.TreeExplainer(lgbm_raw)

LGD_ASSUMED = 0.45

st.title("Credit Default Risk & Expected Credit Loss Estimator")
st.caption(
    "Demo app for a calibrated probability-of-default model trained on the UCI "
    "Default of Credit Card Clients dataset. Enter a hypothetical customer's "
    "profile to see predicted risk, ECL, and the model's reasoning."
)

st.warning(
    "This is a portfolio/demo tool trained on 2005 Taiwan data. It is not "
    "financial advice and should not be used for real lending decisions.",
    icon="⚠️",
)

# ---- Inputs ----
st.header("Customer Profile")

col1, col2 = st.columns(2)
with col1:
    limit_bal = st.number_input("Credit Limit (NT$)", min_value=10000, max_value=1000000, value=150000, step=10000)
    sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    education = st.selectbox("Education", options=[1, 2, 3, 4],
                              format_func=lambda x: {1: "Graduate school", 2: "University", 3: "High school", 4: "Others"}[x])
with col2:
    marriage = st.selectbox("Marital Status", options=[1, 2, 3],
                             format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x])
    age = st.number_input("Age", min_value=18, max_value=90, value=35)

st.subheader("Repayment Status — last 6 months")
st.caption("−1 = paid on time, 1–8 = months overdue. Most recent month first.")
pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
pay_labels = ["Most recent", "2 months ago", "3 months ago", "4 months ago", "5 months ago", "6 months ago"]
pay_values = {}
cols = st.columns(6)
for c, col_name, label in zip(cols, pay_cols, pay_labels):
    with c:
        pay_values[col_name] = st.slider(label, min_value=-1, max_value=8, value=-1, key=col_name)

st.subheader("Bill & Payment Amounts — last 6 months")
bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
payamt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
bill_values, payamt_values = {}, {}
for i in range(6):
    c1, c2 = st.columns(2)
    with c1:
        bill_values[bill_cols[i]] = st.number_input(f"Bill amount, month {i+1}", min_value=0, value=20000, step=1000, key=f"bill_{i}")
    with c2:
        payamt_values[payamt_cols[i]] = st.number_input(f"Amount paid, month {i+1}", min_value=0, value=5000, step=500, key=f"pay_{i}")

# ---- Build feature row ----
raw_row = {
    "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education, "MARRIAGE": marriage, "AGE": age,
    **pay_values, **bill_values, **payamt_values,
}
raw_df = pd.DataFrame([raw_row])
feat_df = add_features(raw_df)

# LightGBM was trained without the 'default' column; ensure column order matches
expected_cols = lgbm_raw.booster_.feature_name()
feat_df = feat_df[expected_cols]

# ---- Predict ----
if st.button("Assess Risk", type="primary"):
    pd_calibrated = lgbm_cal.predict_proba(feat_df)[0, 1]
    ead = limit_bal
    ecl = pd_calibrated * LGD_ASSUMED * ead

    if pd_calibrated < 0.10:
        risk_band = "Low"
    elif pd_calibrated < 0.25:
        risk_band = "Medium"
    elif pd_calibrated < 0.50:
        risk_band = "High"
    else:
        risk_band = "Very High"

    st.header("Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Probability of Default", f"{pd_calibrated:.1%}")
    m2.metric("Risk Band", risk_band)
    m3.metric("Expected Credit Loss", f"NT${ecl:,.0f}")

    st.caption(
        f"ECL = PD ({pd_calibrated:.1%}) × LGD ({LGD_ASSUMED:.0%}, assumed) × "
        f"EAD (NT${ead:,.0f} credit limit)"
    )

    st.subheader("Why this prediction?")
    st.caption("SHAP feature attribution — how each factor pushed this specific prediction toward or away from default.")

    shap_values = explainer.shap_values(feat_df)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

    explanation = shap.Explanation(
        values=sv[0], base_values=base_val, data=feat_df.iloc[0], feature_names=feat_df.columns.tolist()
    )
    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation, show=False, max_display=10)
    st.pyplot(fig)

    st.info(
        "This model was calibrated on historical data — predicted probabilities matched "
        "actual observed default rates to within thousandths across risk bands during "
        "evaluation. See the full methodology on GitHub.",
        icon="✅",
    )

st.divider()
st.caption(
    "[Full project & methodology on GitHub](https://github.com/Edgezone-commits/credit-risk-pd-ecl) · "
    "Trained on UCI Default of Credit Card Clients dataset (Yeh, 2009, CC BY 4.0)"
)