import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. LOAD THE MODEL ---
# Make sure 'heart_disease_model.pkl' is in the same folder as this script!
try:
    model = joblib.load('heart_disease_model.pkl')
    # Get the features the model was trained on
    model_features = model.feature_names_in_
except:
    st.error("Model file not found. Please ensure 'heart_disease_model.pkl' is in the folder.")

# --- 2. APP INTERFACE ---
st.set_page_config(page_title="Heart Stroke Predictor", page_icon="❤️")

st.title("❤️ Heart Stroke Risk Assessment")
st.markdown("""
This AI tool uses clinical markers to estimate the risk of stroke. 
*Iterative Model Version 2.0 (Clinical Focus)*
""")

st.sidebar.header("Patient Data Input")

def get_user_input():
    # Basic Demographics
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=45)
    gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
    
    # Vital Signs
    sysBP = st.sidebar.number_input("Systolic Blood Pressure (sysBP)", min_value=70, max_value=250, value=120)
    diaBP = st.sidebar.number_input("Diastolic Blood Pressure (diaBP)", min_value=40, max_value=150, value=80)
    heartRate = st.sidebar.number_input("Heart Rate", min_value=40, max_value=200, value=75)
    
    # Lab Results
    totChol = st.sidebar.number_input("Total Cholesterol", min_value=100, max_value=600, value=200)
    glucose = st.sidebar.number_input("Glucose Level", min_value=40, max_value=500, value=100)
    bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

    # --- 3. HANDLE ITERATIVE FEATURES ---
    # We must calculate pulse_pressure because the model expects it!
    pulse_pressure = sysBP - diaBP
    
    # Handle the 'male_1' or 'male_True' dummy variable
    is_male = 1 if gender == "Male" else 0

    # Create the dictionary to match the training columns EXACTLY
    data = {
        'age': age,
        'totChol': totChol,
        'sysBP': sysBP,
        'diaBP': diaBP,
        'BMI': bmi,
        'heartRate': heartRate,
        'glucose': glucose,
        'pulse_pressure': pulse_pressure,
        'male_1': is_male # Change this to 'male_True' if your dummies used booleans
    }
    
    return pd.DataFrame([data])

input_df = get_user_input()

# --- 4. PREDICTION ---
if st.button("Calculate Stroke Risk"):
    # Ensure columns are in the exact same order as training
    input_df = input_df[model_features]
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Result:")
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK: The model predicts a high risk of stroke.")
    else:
        st.success(f"✅ LOW RISK: The model predicts a low risk of stroke.")
        
    st.info(f"Probability Score: {probability:.2%}")
    
    # Contextual Disclaimer
    st.caption("**Disclaimer:** This is a prototype AI tool for educational purposes. Always consult a medical professional.")