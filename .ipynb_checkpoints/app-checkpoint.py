import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("ebm_model.pkl")

st.title("Cardiovascular Risk Predictor")

st.write("Enter patient details:")

# Example input fields (adjust to your dataset features)
age = st.number_input("Age", min_value=1, max_value=120)
bmi = st.number_input("BMI")
systolic_bp = st.number_input("Systolic Blood Pressure")
diastolic_bp = st.number_input("Diastolic Blood Pressure")
cholesterol = st.number_input("Cholesterol Level")
glucose = st.number_input("Glucose Level")
smoker = st.selectbox("Smoker", [0, 1])
alcohol = st.selectbox("Alcohol Intake", [0, 1])
physical_activity = st.selectbox("Physical Activity", [0, 1])

# Predict button
if st.button("Predict Risk"):
    # Arrange features exactly as used in training
    features = np.array([[age, bmi, systolic_bp, diastolic_bp,
                          cholesterol, glucose, smoker,
                          alcohol, physical_activity]])

    prob = model.predict_proba(features)[0][1]

    st.subheader(f"Predicted Cardiovascular Risk: {prob:.2%}")

    if prob > 0.5:
        st.error("High Risk ⚠️")
    else:
        st.success("Low Risk ✅")