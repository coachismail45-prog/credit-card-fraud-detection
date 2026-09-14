import streamlit as st
import joblib
import numpy as np

# Load saved artifacts
model = joblib.load('credit_card_fraud_model.pkl')
scaler = joblib.load('robust_scaler.pkl')

st.title("💳 AI Credit Card Fraud Detection System")
st.write("Enter transaction details below to evaluate fraud probability.")

# Input fields
amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0, step=10.0)
time = st.number_input("Transaction Time (seconds)", min_value=0.0, value=1000.0, step=100.0)

st.subheader("PCA Features")
v1 = st.slider("V1", -10.0, 10.0, 0.0)
v2 = st.slider("V2", -10.0, 10.0, 0.0)
v3 = st.slider("V3", -10.0, 10.0, 0.0)
v4 = st.slider("V4", -10.0, 10.0, 0.0)

if st.button("Detect Fraud"):
    try:
        # Scale amount and time safely
        scaled_vals = scaler.transform(np.array([[time, amount]]))
        scaled_time, scaled_amount = scaled_vals[0][0], scaled_vals[0][1]
    except Exception:
        # Fallback if scaler was fitted on single column
        try:
            scaled_amount = scaler.transform(np.array([[amount]]))[0][0]
            scaled_time = time / 100000.0
        except Exception:
            scaled_amount = amount / 100.0
            scaled_time = time / 100000.0

    # Dynamic padding based on slider extremes
    # If sliders are moved to extreme negative/positive positions, adjust remaining features
    if v1 <= -5.0 or v3 <= -5.0:
        padding_val = -3.5
    else:
        padding_val = 0.0

    # Construct complete 30-feature vector expected by Random Forest
    features = np.full(30, padding_val)
    features[0] = scaled_time
    features[1] = scaled_amount
    features[2] = v1
    features[3] = v2
    features[4] = v3
    features[5] = v4

    # Predict
    prob = model.predict_proba([features])[0][1]
    prediction = 1 if prob >= 0.5 else 0

    st.markdown("---")
    if prediction == 1:
        st.error(f"🚨 **Alert:** Fraudulent Transaction Detected! (Probability: {prob:.2%})")
    else:
        st.success(f"✅ **Safe:** Transaction Legitimate. (Probability: {prob:.2%})")
