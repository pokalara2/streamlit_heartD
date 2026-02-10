import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. PAGE CONFIG & STYLING ---
st.set_page_config(page_title="Heart Stroke Risk Assistant", page_icon="❤️", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD THE MODEL ---
@st.cache_resource
def load_model():
    # Ensure this filename matches your uploaded .pkl file exactly
    model = joblib.load('heart_disease_model.pkl')
    return model

try:
    model = load_model()
    model_features = model.feature_names_in_
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 3. APP HEADER ---
st.title("❤️ Heart Stroke Risk Assessment")
st.markdown("""
    This AI prototype identifies potential stroke risk based on clinical markers. 
    **Designed for screening assistance.**
""")
st.divider()

# --- 4. SIDEBAR INPUTS ---
st.sidebar.header("📋 Patient Profile")

def get_user_input():
    # Demographics
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=45)
    gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
    
    # Lifestyle
    currentSmoker = st.sidebar.selectbox("Do you smoke?", options=["No", "Yes"])
    cigsPerDay = st.sidebar.number_input("Cigarettes per day (if smoker)", min_value=0, value=0) if currentSmoker == "Yes" else 0
    
    # Medical History
    st.sidebar.subheader("Medical History")
    BPMeds = st.sidebar.selectbox("Are you on Blood Pressure Medication?", options=["No", "Yes"])
    prevalentHyp = st.sidebar.selectbox("Do you have Hypertension?", options=["No", "Yes"])
    diabetes = st.sidebar.selectbox("Do you have Diabetes?", options=["No", "Yes"])
    prevalentStroke = st.sidebar.selectbox("Have you ever had a stroke?", options=["No", "Yes"])

    # Clinical Vitals (with Help Text for UX)
    st.sidebar.subheader("Clinical Vitals")
    sysBP = st.sidebar.number_input(
        "Systolic Blood Pressure (Top Number)", 
        value=120, 
        help="Normal is ~120. High is 140+."
    )
    diaBP = st.sidebar.number_input(
        "Diastolic Blood Pressure (Bottom Number)", 
        value=80, 
        help="Normal is ~80. High is 90+."
    )
    
    # Blood Work & BMI
    totChol = st.sidebar.number_input("Total Cholesterol", value=200)
    glucose = st.sidebar.number_input("Glucose Level", value=100)
    bmi = st.sidebar.number_input("BMI (Body Mass Index)", value=25.0, format="%.1f")
    heartRate = st.sidebar.number_input("Heart Rate (BPM)", value=75)

    # Reference Guide for non-medical users
    with st.sidebar.expander("🩺 Blood Pressure Guide"):
        st.write("**Normal:** 120/80")
        st.write("**Elevated:** 130/80")
        st.write("**High (Stage 1):** 140/90")
        st.write("**High (Stage 2):** 160/100+")

    # --- 5. DATA TRANSFORMATION ---
    # Convert text to binary (1/0) and match model columns exactly
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

# --- 6. PREDICTION & RESULTS ---
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Analyze Risk Profile"):
        # Match feature order
        final_input = input_df[model_features]
        
        # Get Probability
        probability = model.predict_proba(final_input)[0][1]

        st.subheader("Risk Assessment:")
        
        if probability < 0.30:
            st.success(f"### 🟢 LOW RISK ({probability:.1%})")
            st.write("Your clinical markers suggest a low current risk. Maintain a healthy lifestyle and routine checkups.")
            
        elif 0.30 <= probability < 0.60:
            st.warning(f"### 🟡 MEDIUM RISK ({probability:.1%})")
            st.write("Some clinical indicators are elevated. We recommend discussing this report with a healthcare provider.")
            
        else:
            st.error(f"### 🔴 HIGH RISK ({probability:.1%})")
            st.markdown("#### 🚨 ACTION REQUIRED")
            st.write("This profile strongly correlates with high-risk stroke data. Please seek a professional medical evaluation soon.")

        st.progress(probability)

with col2:
    st.info("💡 **How it works**\nOur Random Forest model analyzes your vitals against thousands of patient records to find patterns linked to stroke events.")

# --- 7. FOOTER & LEGAL ---
st.divider()
st.warning("""
**⚠️ Medical Disclaimer:** This application is an AI prototype for educational and screening purposes only. 
It is **not** a diagnostic tool. AI can produce errors. If you are experiencing symptoms like facial drooping, 
arm weakness, or speech difficulty, call emergency services immediately.
""")
