from __future__ import annotations

import pandas as pd
import streamlit as st

from churn import artifacts, data

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

YES_NO = ["No", "Yes"]
ADDON_LABELS = {
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
}
PAYMENT_METHODS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


@st.cache_resource(show_spinner=False)
def load_artifacts():
    return artifacts.load_model()


try:
    pipeline, card = load_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

threshold = float(card["threshold"])

st.title("Telco customer churn predictor")
st.caption(
    f"Model: **{card['model']}** ({card['imbalance_strategy']}) · "
    f"operating threshold **{threshold:.2f}** · trained {card['trained_at_utc'][:10]}"
)

col_customer, col_services, col_billing, col_result = st.columns([1, 1, 1, 1.1], gap="large")

with col_customer:
    st.subheader("Customer")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", YES_NO)
    partner = st.selectbox("Has partner", YES_NO)
    dependents = st.selectbox("Has dependents", YES_NO)
    tenure = st.slider("Tenure (months)", 0, 72, 12)

with col_services:
    st.subheader("Services")
    phone = st.selectbox("Phone service", ["Yes", "No"])
    if phone == "No":
        multiple_lines = st.selectbox("Multiple lines", ["No phone service"], disabled=True)
    else:
        multiple_lines = st.selectbox("Multiple lines", YES_NO)
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    addons = {}
    for col, label in ADDON_LABELS.items():
        options = ["No internet service"] if internet == "No" else YES_NO
        addons[col] = st.selectbox(label, options, disabled=internet == "No")

with col_billing:
    st.subheader("Contract & billing")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox("Payment method", PAYMENT_METHODS)
    monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, step=0.5)
    total_override = st.number_input(
        "Total charges to date ($)",
        0.0,
        10000.0,
        0.0,
        step=10.0,
        help="Leave at 0 to estimate as tenure x monthly charges.",
    )

total_charges = total_override if total_override > 0 else tenure * monthly

record = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple_lines,
    "InternetService": internet,
    **addons,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total_charges,
}

features = data.prepare_features(pd.DataFrame([record]))
probability = float(pipeline.predict_proba(features)[0, 1])

if probability >= threshold:
    band, render = "High risk", st.error
elif probability >= threshold / 2:
    band, render = "Medium risk", st.warning
else:
    band, render = "Low risk", st.success

with col_result:
    st.subheader("Prediction")
    st.metric("Churn probability", f"{probability:.1%}")
    st.progress(min(max(probability, 0.0), 1.0))
    render(f"**{band}** — flagged as churner: **{'Yes' if probability >= threshold else 'No'}**")
    st.caption(
        f"A customer is flagged when probability ≥ {threshold:.2f}. "
        "Probabilities come from a model trained on a sample dataset and are for demonstration only."
    )
    with st.expander("Model card"):
        st.json(card)
