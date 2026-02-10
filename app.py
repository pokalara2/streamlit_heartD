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
    # Keep your existing inputs (age, sysBP, etc.)
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=45)
    gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
    
    # NEW: Add the missing categorical inputs
    currentSmoker = st.sidebar.selectbox("Do you smoke?", options=["Yes", "No"])
    cigsPerDay = st.sidebar.number_input("Cigarettes per day", value=0) if currentSmoker == "Yes" else 0
    BPMeds = st.sidebar.selectbox("Are you on Blood Pressure meds?", options=["Yes", "No"])
    prevalentHyp = st.sidebar.selectbox("Do you have Hypertension?", options=["Yes", "No"])
    diabetes = st.sidebar.selectbox("Do you have Diabetes?", options=["Yes", "No"])
    prevalentStroke = st.sidebar.selectbox("Have you had a stroke before?", options=["Yes", "No"])

    # Existing lab values
    sysBP = st.sidebar.number_input("Systolic BP", value=120)
    diaBP = st.sidebar.number_input("Diastolic BP", value=80)
    totChol = st.sidebar.number_input("Total Cholesterol", value=200)
    bmi = st.sidebar.number_input("BMI", value=25.0)
    heartRate = st.sidebar.number_input("Heart Rate", value=75)
    glucose = st.sidebar.number_input("Glucose", value=100)

    # --- MATCHING THE MODEL'S EXPECTED NAMES ---
    data = {
        'age': age,
        'totChol': totChol,
        'sysBP': sysBP,
        'diaBP': diaBP,
        'BMI': bmi,
        'heartRate': heartRate,
        'glucose': glucose,
        'pulse_pressure': sysBP - diaBP,
        'currentSmoker': 1 if currentSmoker == "Yes" else 0,
        'cigsPerDay': cigsPerDay,
        'BPMeds': 1 if BPMeds == "Yes" else 0,
        'prevalentHyp': 1 if prevalentHyp == "Yes" else 0,
        'diabetes': 1 if diabetes == "Yes" else 0,
        'Gender_Male': 1 if gender == "Male" else 0,
        'prevalentStroke_yes': 1 if prevalentStroke == "Yes" else 0
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
