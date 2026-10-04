import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# FIX: Page config must be the FIRST streamlit command right after imports
st.set_page_config(page_title="MediAssist_AI", layout="wide")

# ==========================================
# 1. MODEL TRAINING & DATA LOADING (Background)
# ==========================================
@st.cache_resource
def load_model_and_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # Selecting top 5 features for a clean Web UI
    selected_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean concavity']
    X_sel = X[selected_features]
    
    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Ensemble Model (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, selected_features, X_train_scaled

model, scaler, selected_features, X_train_bg = load_model_and_data()
explainer = shap.TreeExplainer(model)

# ==========================================
# 2. STREAMLIT UI LAYOUT
# ==========================================
st.sidebar.title("🏥 MediAssist AI")
page = st.sidebar.radio("Navigation", ["Home", "Cancer Predictor (XAI)", "Medical NLP Extractor"])

# --- PAGE 1: HOME ---
if page == "Home":
    st.title(" Welcome to MediAssist AI")
    st.markdown("""
    **An Integrated Explainable AI Healthcare Web Application**
    
    This mini-project integrates multiple AI domains to assist healthcare professionals:
    - **🎗️ Cancer Predictor:** Predicts Breast Cancer risk using Ensemble Learning and explains the decision using SHAP (Explainable AI).
    - **📝 Medical NLP:** Extracts diseases and medications from unstructured medical reports using Natural Language Processing.
    
    *Select a module from the sidebar to get started!*
    """)

# --- PAGE 2: CANCER PREDICTOR & XAI ---
elif page == "Cancer Predictor (XAI)":
    st.title("🎗️ Breast Cancer Risk Predictor & Explainer")
    st.write("Enter the top 5 critical tumor features to get a prediction and its AI explanation.")
    
    col1, col2 = st.columns(2)
    inputs = []
    
    # Taking inputs for the 5 selected features
    with col1:
        for feat in selected_features[:3]:
            val = st.number_input(f"Enter {feat}", value=10.0, step=0.1)
            inputs.append(val)
    with col2:
        for feat in selected_features[3:]:
            val = st.number_input(f"Enter {feat}", value=10.0, step=0.1)
            inputs.append(val)
            
    if st.button("🔍 Predict & Explain"):
        # Scale user input
        user_input = np.array(inputs).reshape(1, -1)
        user_input_scaled = scaler.transform(user_input)
        
        # Predict
        prediction = model.predict(user_input_scaled)[0]
        prob = model.predict_proba(user_input_scaled)[0]
        
        st.subheader("Prediction Result:")
        if prediction == 0:
            st.error(f"⚠️ **MALIGNANT (Cancerous)** | Confidence: {prob[0]*100:.2f}%")
        else:
            st.success(f"✅ **BENIGN (Non-Cancerous)** | Confidence: {prob[1]*100:.2f}%")
            
        # Explainable AI (SHAP)
        st.subheader("🧠 AI Explanation (SHAP Force Plot)")
        st.write("The plot below shows how each feature contributed to the final prediction probability (Red = Increasing probability, Blue = Decreasing probability)."
)
        
        # Get raw SHAP values and the predicted class
        shap_values_raw = explainer.shap_values(user_input_scaled)
        prediction_class = model.predict(user_input_scaled)[0]
        
        # Extract strictly 1D array for the specific predicted class (0 or 1)
        if isinstance(shap_values_raw, list):
            sv_to_plot = shap_values_raw[prediction_class][0, :]
            ev_to_plot = explainer.expected_value[prediction_class]
        else:
            sv_to_plot = shap_values_raw[0, :, prediction_class]
            ev_to_plot = explainer.expected_value[prediction_class]

        # Create a modern SHAP Explanation object
        explanation = shap.Explanation(
            values=sv_to_plot,
            base_values=ev_to_plot,
            data=user_input_scaled[0],
            feature_names=selected_features
        )
        
        # Plot using Waterfall
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.title("SHAP Explanation: Feature Impact on Prediction")
        plt.tight_layout()
        st.pyplot(fig)

# --- PAGE 3: MEDICAL NLP ---
elif page == "Medical NLP Extractor":
    st.title(" Medical Report Entity Extractor")
    st.write("Paste an unstructured medical report below to automatically extract key entities using NLP.")
    
    sample_text = "Patient John, 55, diagnosed with hypertension and type 2 diabetes. Prescribed Metformin and Aspirin."
    text = st.text_area("Enter Medical Report:", value=sample_text, height=150)
    
    if st.button("🔎 Extract Entities"):
        # Custom Dictionary based NLP
        diseases = ["cancer", "diabetes", "hypertension", "asthma", "tumor", "pneumonia", "infarction"]
        meds = ["aspirin", "metformin", "insulin", "paracetamol", "ibuprofen", "lisinopril"]
        
        text_lower = text.lower()
        
        found_diseases = [d.title() for d in diseases if d in text_lower]
        found_meds = [m.title() for m in meds if m in text_lower]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🦠 Diseases/Symptoms Found")
            st.write(", ".join(found_diseases) if found_diseases else "None")
        with col2:
            st.subheader(" Medications Found")
            st.write(", ".join(found_meds) if found_meds else "None")