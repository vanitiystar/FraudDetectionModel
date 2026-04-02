import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Load model and scaler ──────────────────────────────────────────────────
model  = joblib.load('fraud_ensemble.pkl')
scaler = joblib.load('fraud_scaler.pkl')

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Fraud Detection System",
    page_icon  = "🔍",
    layout     = "centered"
)

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🔍 Mobile Transaction Fraud Detector")
st.markdown("Enter the transaction details below to check if it is fraudulent.")
st.divider()

# ── Input form ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    txn_type     = st.selectbox("Transaction Type",  ["TRANSFER", "CASH_OUT"])
    amount       = st.number_input("Transaction Amount",        min_value=0.0, value=50000.0, step=1000.0)
    old_bal_orig = st.number_input("Sender Old Balance",        min_value=0.0, value=100000.0, step=1000.0)
    new_bal_orig = st.number_input("Sender New Balance",        min_value=0.0, value=50000.0,  step=1000.0)

with col2:
    step         = st.number_input("Step (hour of transaction)", min_value=1, max_value=744, value=1)
    old_bal_dest = st.number_input("Receiver Old Balance",      min_value=0.0, value=10000.0, step=1000.0)
    new_bal_dest = st.number_input("Receiver New Balance",      min_value=0.0, value=60000.0, step=1000.0)

st.divider()

# ── Predict button ─────────────────────────────────────────────────────────
if st.button("Check Transaction", use_container_width=True, type="primary"):

    # Feature engineering — same as training
    balance_diff_orig = old_bal_orig - new_bal_orig
    balance_diff_dest = new_bal_dest - old_bal_dest
    balance_mismatch  = balance_diff_orig - amount
    orig_zeroed_out   = 1 if new_bal_orig == 0 else 0
    amount_ratio_orig = amount / (old_bal_orig + 1)
    type_encoded      = 1 if txn_type == "TRANSFER" else 0

    # Build feature row in exact same order as training
    features = pd.DataFrame([[
        step, amount, old_bal_orig, new_bal_orig,
        old_bal_dest, new_bal_dest,
        balance_diff_orig, balance_diff_dest,
        balance_mismatch, orig_zeroed_out,
        amount_ratio_orig, type_encoded
    ]], columns=[
        'step', 'amount', 'oldbalanceOrig', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest',
        'balance_diff_orig', 'balance_diff_dest',
        'balance_mismatch', 'orig_zeroed_out',
        'amount_ratio_orig', 'type_encoded'
    ])

    # Scale and predict
    features_scaled = scaler.transform(features)
    prob            = model.predict_proba(features_scaled)[0][1]
    verdict         = model.predict(features_scaled)[0]

    # ── Results ────────────────────────────────────────────────────────────
    st.subheader("Results")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Fraud Probability", f"{prob * 100:.2f}%")
    col_b.metric("Verdict",           "FRAUD" if verdict == 1 else "LEGITIMATE")
    col_c.metric("Risk Level",
                 "HIGH"   if prob >= 0.8 else
                 "MEDIUM" if prob >= 0.5 else
                 "LOW")

    st.divider()

    # Advice box
    if verdict == 1:
        if prob >= 0.8:
            st.error("""
            **HIGH RISK — Block this transaction immediately**
            - Do not process this transaction
            - Flag the origin account for investigation
            - Notify the account holder
            """)
        else:
            st.warning("""
            **MEDIUM RISK — Manual review recommended**
            - Hold this transaction for review
            - Contact the account holder to verify
            - Monitor the account for further activity
            """)
    else:
        st.success("""
        **LOW RISK — Transaction appears legitimate**
        - This transaction shows no fraud indicators
        - Safe to process normally
        """)

    # Feature breakdown
    with st.expander("See feature breakdown"):
        st.dataframe(features, use_container_width=True)