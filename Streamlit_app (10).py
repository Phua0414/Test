import streamlit as st
import pickle
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

#Function to download files from GitHub
def download_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception("Error downloading file")

# GitHub URLs for the model and scaler files
model_url = 'https://github.com/Phua0414/Test/releases/download/Tag1/all_models.pkl'
scaler_url = 'https://github.com/Phua0414/Test/releases/download/Tag1/scaler.pkl'

# Download the model and scaler files
model_data = download_file(model_url)
scaler_data = download_file(scaler_url)

# Load the models and scaler from the downloaded data
models = pickle.loads(model_data)
scaler = pickle.loads(scaler_data)

# List of available models
model_names = list(models.keys())

# Feature Names
feature_names = [
    'Gender', 'Hypertension', 'Heart Disease', 'Smoking History',
    'HbA1c Level', 'Blood Glucose Level', 'Age Group', 'BMI Category'
]

# Streamlit App
st.set_page_config(layout="centered")
st.title("🩺 Diabetes Prediction System")

st.header("Select Model")
model_choice = st.selectbox("Choose a Model", model_names)

# User Input Form
st.header("Enter Patient Information")
with st.form("patient_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    smoking_history = st.selectbox("Smoking History", ["never", "No Info", "current", "former", "ever", "not current"])
    hba1c = st.slider("HbA1c Level", 4.0, 10.0, 6.0)
    glucose = st.slider("Blood Glucose Level", 50, 300, 100)
    age = st.number_input("Age", 18, 100, value=25)
    bmi = st.number_input("BMI", 10.0, 50.0, value=25.0)

    submit = st.form_submit_button("🔮 Predict")

if submit:
    # 🧠 Feature Engineering
    gender = 0 if gender == "Male" else 1
    hypertension = 0 if hypertension == "No" else 1
    heart_disease = 0 if heart_disease == "No" else 1
    smoking_mapping = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5}
    smoking_history = smoking_mapping[smoking_history]

    age_group = 0 if age < 18 else 1 if age < 25 else 2 if age < 45 else 3 if age < 60 else 4
    bmi_category = 0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3

    input_features = np.array([[gender, hypertension, heart_disease, smoking_history, hba1c, glucose, age_group, bmi_category]])

    # 🧹 Scale and Predict
    scaled_input = scaler.transform(input_features)
    model = models[model_choice]['model']

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    # 🧠 Show Prediction
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"Prediction: **Diabetic** (Risk: {probability*100:.1f}%)")
        st.warning("⚠️ High likelihood of diabetes. Please consult a doctor.")
    else:
        st.success(f"Prediction: **Not Diabetic** (Risk: {probability*100:.1f}%)")
        st.info("✅ Low likelihood, but maintain healthy habits!")

    # 🩺 Risk Interpretation
    st.subheader("Risk Interpretation")
    if probability < 0.3:
        st.success("🟢 Low Risk (0% - 30%)")
    elif probability < 0.7:
        st.warning("🟠 Moderate Risk (30% - 70%)")
    else:
        st.error("🔴 High Risk (70% - 100%)")
