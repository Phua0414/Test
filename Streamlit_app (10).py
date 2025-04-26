import streamlit as st
import pickle
import numpy as np
import requests

# Download files from GitHub
def download_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception("Error downloading file")

# GitHub URLs
model_url = 'https://github.com/Phua0414/Test/releases/download/Tag-1/all_models.pkl'
scaler_url = 'https://github.com/Phua0414/Test/releases/download/Tag-1/scaler.pkl'

# Load model and scaler
model_data = download_file(model_url)
scaler_data = download_file(scaler_url)
models = pickle.loads(model_data)
scaler = pickle.loads(scaler_data)

# Streamlit UI Setup
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered", page_icon="🩺")
st.title("🩺 Diabetes Prediction System")

# Sidebar for model selection
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose a Model", list(models.keys()))

# User Inputs
st.header("🧍 Patient Information")

with st.form("patient_form"):
    st.subheader("📋 Demographic Information")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 18, 100, value=25, help="Enter age in years")
    with col2:
        bmi = st.number_input("BMI", 10.0, 50.0, value=25.0, help="Body Mass Index")

    st.subheader("🩺 Medical History")
    col3, col4 = st.columns(2)
    with col3:
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    with col4:
        smoking_history = st.selectbox(
            "Smoking History",
            ["never", "No Info", "current", "former", "ever", "not current"],
            help="Past or present smoking habits"
        )

    st.subheader("🧪 Lab Results")
    col5, col6 = st.columns(2)
    with col5:
        hba1c = st.slider("HbA1c Level", 4.0, 10.0, 6.0, help="Average blood sugar over past 3 months")
    with col6:
        glucose = st.slider("Blood Glucose Level", 50, 300, 100, help="Current glucose reading")

    submit = st.form_submit_button("🔮 Predict Diabetes")

# Prediction Logic
if submit:
    gender = 0 if gender == "Male" else 1
    hypertension = 0 if hypertension == "No" else 1
    heart_disease = 0 if heart_disease == "No" else 1
    smoking_mapping = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5}
    smoking_history = smoking_mapping[smoking_history]

    # Encode age group & BMI category
    age_group = 0 if age < 18 else 1 if age < 25 else 2 if age < 45 else 3 if age < 60 else 4
    bmi_category = 0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3

    input_features = np.array([[gender, hypertension, heart_disease, smoking_history, hba1c, glucose, age_group, bmi_category]])
    scaled_input = scaler.transform(input_features)

    model = models[model_choice]
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"🚨 Prediction: **Diabetic**\n\n🩸 Risk: **{probability*100:.1f}%**")
        st.warning("⚠️ High likelihood of diabetes. Please consult a healthcare provider.")
    else:
        st.success(f"🟢 Prediction: **Not Diabetic**\n\n✅ Risk: **{probability*100:.1f}%**")
        st.info("Keep up with healthy habits and regular checkups!")

    # Risk Level Bar
    st.subheader("📊 Risk Level Breakdown")
    if probability < 0.3:
        st.success("🟢 Low Risk (0% - 30%)")
    elif probability < 0.7:
        st.warning("🟠 Moderate Risk (30% - 70%)")
    else:
        st.error("🔴 High Risk (70% - 100%)")
