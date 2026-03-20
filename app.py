import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Customer Churn Analytics", page_icon="📊", layout="wide")

# Custom Title
st.title("📊 Customer Churn Analytics & Prediction Dashboard")
st.markdown("""
This dashboard provides comprehensive insights into customer behavior and a predictive model 
to identify customers at high risk of churning. Use the sidebar to navigate between data exploration and prediction.
""")

# Load Data
@st.cache_data
def load_data():
    if os.path.exists('customer_churn_clean.csv'):
        return pd.read_csv('customer_churn_clean.csv')
    return pd.DataFrame()

df = load_data()

# Load Model Artifacts
@st.cache_resource
def load_model():
    try:
        model = joblib.load('churn_rf_model.pkl')
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, scaler, model_columns
    except FileNotFoundError:
        return None, None, None

rf_model, scaler, model_columns = load_model()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Exploratory Data Analysis", "Predict Churn"])

if page == "Dashboard Overview":
    st.header("Key Performance Indicators (KPIs)")
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        total_customers = len(df)
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        avg_tenure = df['Tenure'].mean()
        avg_monthly_charges = df['MonthlyCharges'].mean()
        
        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
        col3.metric("Avg. Tenure (Months)", f"{avg_tenure:.1f}")
        col4.metric("Avg. Monthly Charges", f"${avg_monthly_charges:.2f}")
        
        st.subheader("Recent Customer Data")
        st.dataframe(df.head(10))
        
        st.subheader("Business Impact")
        st.markdown("""
        **Why Customer Churn Matters:**
        - Acquiring a new customer is 5-25x more expensive than retaining an existing one.
        - Increasing customer retention rates by 5% increases profits by 25% to 95%.
        - With a churn rate of **{:.1f}%**, identifying at-risk customers early is critical for proactive intervention.
        """.format(churn_rate))
    else:
        st.warning("Data not found. Please ensure the data generation and analysis scripts have been run.")

elif page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    st.markdown("Discover the factors driving customer churn.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Distribution")
        if os.path.exists("images/churn_distribution.png"):
            st.image("images/churn_distribution.png", use_column_width=True)
            st.markdown("- **Insight:** The dataset is somewhat imbalanced, which we addressed during modeling using class weights.")
            
        st.subheader("Monthly Charges vs Churn")
        if os.path.exists("images/monthly_charges_vs_churn.png"):
            st.image("images/monthly_charges_vs_churn.png", use_column_width=True)
            st.markdown("- **Insight:** Customers with higher monthly charges show a higher propensity to churn.")
            
    with col2:
        st.subheader("Tenure Distribution by Churn")
        if os.path.exists("images/tenure_vs_churn.png"):
            st.image("images/tenure_vs_churn.png", use_column_width=True)
            st.markdown("- **Insight:** New customers (low tenure) are highly vulnerable to churning. Retention efforts should focus on early onboarding.")
            
        st.subheader("Feature Importance")
        if os.path.exists("images/feature_importance.png"):
            st.image("images/feature_importance.png", use_column_width=True)
            st.markdown("- **Insight:** Tenure, Total Charges, and Monthly Charges are the most critical predictors of churn according to the Random Forest model.")

elif page == "Predict Churn":
    st.header("Predictive Modeling: Churn Risk Assessment")
    st.markdown("Enter customer details to predict their likelihood of churning.")
    
    if rf_model is not None:
        with st.form("churn_prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior_citizen = st.selectbox("Senior Citizen", [0, 1])
                tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
                
            with col2:
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                
            with col3:
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=75.0)
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=20000.0, value=monthly_charges * tenure)
                
            submit_button = st.form_submit_button(label="Predict Churn Risk")
            
        if submit_button:
            # Prepare input data
            input_data = {
                'Gender': [1 if gender == "Male" else 0],  # Assuming LabelEncoder encoded Female=0, Male=1 alphabetically
                'SeniorCitizen': [senior_citizen],
                'Tenure': [tenure],
                'PaperlessBilling': [1 if paperless_billing == "Yes" else 0],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges],
                'Contract_One year': [1 if contract == "One year" else 0],
                'Contract_Two year': [1 if contract == "Two year" else 0],
                'PaymentMethod_Credit card (automatic)': [1 if payment_method == "Credit card (automatic)" else 0],
                'PaymentMethod_Electronic check': [1 if payment_method == "Electronic check" else 0],
                'PaymentMethod_Mailed check': [1 if payment_method == "Mailed check" else 0]
            }
            
            input_df = pd.DataFrame(input_data)
            
            # Reorder columns to match model training
            for col in model_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[model_columns]
            
            # Scale features
            features_to_scale = ['Tenure', 'MonthlyCharges', 'TotalCharges']
            input_df[features_to_scale] = scaler.transform(input_df[features_to_scale])
            
            # Predict
            prediction = rf_model.predict(input_df)[0]
            probability = rf_model.predict_proba(input_df)[0][1]
            
            st.markdown("---")
            st.subheader("Prediction Results")
            
            if prediction == 1:
                st.error(f"⚠️ **High Risk of Churn!** (Probability: {probability:.1%})")
                st.markdown("**Recommendation:** Proactively reach out to this customer with targeted retention offers (e.g., discounts, upgrades).")
            else:
                st.success(f"✅ **Low Risk of Churn.** (Probability: {probability:.1%})")
                st.markdown("**Recommendation:** Continue providing excellent service.")
    else:
        st.error("Model not found. Please run the analysis script to train and save the model.")
