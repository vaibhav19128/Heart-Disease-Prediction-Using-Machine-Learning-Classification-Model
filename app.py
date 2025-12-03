
import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

st.set_page_config(page_title="Heart Disease Prediction")

st.title("❤️ Heart Disease Prediction App")
st.write("Predict heart disease using a trained Random Forest model (example).")

# Check models exist
if not os.path.exists("models/scaler.pkl") or not os.path.exists("models/random_forest.pkl"):
    st.warning("Trained models not found. Run `python train.py` first to generate models.")
else:
    scaler = joblib.load("models/scaler.pkl")
    rf_model = joblib.load("models/random_forest.pkl")

    with st.form("input_form"):
        age = st.number_input("Age", 1, 120, value=45)
        sex = st.selectbox("Sex (1=Male, 0=Female)", [1,0], index=0)
        cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3], index=1)
        trestbps = st.number_input("Resting Blood Pressure", 50, 250, value=120)
        chol = st.number_input("Cholesterol", 50, 600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 (1=True, 0=False)", [0,1], index=0)
        restecg = st.selectbox("Rest ECG (0-2)", [0,1,2], index=0)
        thalach = st.number_input("Max Heart Rate Achieved", 50, 250, value=150)
        exang = st.selectbox("Exercise Induced Angina (0/1)", [0,1], index=0)
        oldpeak = st.number_input("ST Depression induced by exercise", 0.0, 10.0, value=1.0)
        slope = st.selectbox("Slope (0-2)", [0,1,2], index=1)
        ca = st.selectbox("Number of major vessels (0-4)", [0,1,2,3,4], index=0)
        thal = st.selectbox("Thal (0-3)", [0,1,2,3], index=1)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]])

        try:
            input_scaled = scaler.transform(input_data)
            rf_pred = rf_model.predict(input_scaled)[0]
            st.subheader("Prediction Result:")
            if rf_pred == 1:
                st.success("Heart Disease: YES")
            else:
                st.info("Heart Disease: NO")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
