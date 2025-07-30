import streamlit as st
import joblib
import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import preprocess_telco_data

@st.cache_resource
def load_model():
    return joblib.load('models/VotingCLF.pkl')

model = load_model()

st.title("🗳️ VotingCLF - Telco Churn Prediction")
st.markdown("---")

st.markdown("""
This model predicts customer churn based on demographic information, 
account details, and service usage patterns using a Voting Classifier ensemble.
""")

st.sidebar.header("Test Cases")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("Low Risk Test", type="secondary", use_container_width=True):
        st.session_state.test_case = "low_risk"

with col2:
    if st.button("High Risk Test", type="secondary", use_container_width=True):
        st.session_state.test_case = "high_risk"

st.sidebar.markdown("---")
st.sidebar.header("Model Information")

with st.sidebar.expander("Individual Model Strengths", expanded=False):
    st.markdown("""
    **Naive Bayes**:
    - 📊 Fast and efficient
    - 🔢 Handles categorical data well
    - 📈 Good baseline performance
    
    **AdaBoost**:
    - 🎯 Focuses on difficult cases
    - 🔄 Iteratively improves predictions
    - ⚖️ Reduces bias and variance
    
    **Support Vector Machine**:
    - 🛡️ Robust against overfitting
    - 📐 Effective in high dimensions
    - 🎪 Handles complex patterns
    
    **Logistic Regression**:
    - 📊 Interpretable coefficients
    - ⚡ Fast training and prediction
    - 🎲 Provides probability estimates
    """)

with st.sidebar.expander("Key Features", expanded=False):
    st.markdown("""
    **Most Important Factors**:
    - 📅 **Tenure**: Customer loyalty duration
    - 💰 **Monthly Charges**: Cost sensitivity
    - 📋 **Contract Type**: Commitment level
    - 🌐 **Internet Service**: Service dependency
    - 💳 **Payment Method**: Payment reliability
    
    **Risk Indicators**:
    - ⚠️ Short tenure (< 12 months)
    - ⚠️ Month-to-month contracts
    - ⚠️ High monthly charges
    - ⚠️ Electronic check payments
    """)

with st.sidebar.expander("Model Performance", expanded=False):
    st.markdown("""
    **Accuracy**: ~94% on test data
    
    **Strengths**:
    - ✅ High precision for churn detection
    - ✅ Balanced performance across customer types
    - ✅ Robust against outliers
    
    **Best Used For**:
    - 🎯 Identifying at-risk customers
    - 📈 Retention campaign targeting
    - 💡 Customer lifetime value analysis
    """)

with st.sidebar.expander("Tips for Best Results", expanded=False):
    st.markdown("""
    **Data Quality**:
    - Ensure accurate customer information
    - Update monthly charges regularly
    - Verify contract details
    
    **Threshold Adjustment**:
    - 🔴 **High Threshold (0.7+)**: Conservative, fewer false alarms
    - 🟡 **Medium Threshold (0.5)**: Balanced approach
    - 🟢 **Low Threshold (0.3-)**: Aggressive, catch more at-risk customers
    """)

st.sidebar.markdown("---")

if 'test_case' not in st.session_state:
    st.session_state.test_case = None

if st.session_state.test_case == "low_risk":
    default_gender = "Female"
    default_senior = "No"
    default_partner = "Yes"
    default_dependents = "Yes"
    default_tenure = 45
    default_charges = 45.0
    default_phone = "Yes"
    default_multiple = "No"
    default_internet = "DSL"
    default_security = "Yes"
    default_backup = "Yes"
    default_protection = "Yes"
    default_support = "Yes"
    default_tv = "No"
    default_movies = "No"
    default_contract = "Two year"
    default_paperless = "No"
    default_payment = "Bank transfer (automatic)"
elif st.session_state.test_case == "high_risk":
    default_gender = "Male"
    default_senior = "Yes"
    default_partner = "No"
    default_dependents = "No"
    default_tenure = 2
    default_charges = 85.0
    default_phone = "Yes"
    default_multiple = "No"
    default_internet = "Fiber optic"
    default_security = "No"
    default_backup = "No"
    default_protection = "No"
    default_support = "No"
    default_tv = "Yes"
    default_movies = "Yes"
    default_contract = "Month-to-month"
    default_paperless = "Yes"
    default_payment = "Electronic check"
else:
    default_gender = "Female"
    default_senior = "No"
    default_partner = "No"
    default_dependents = "No"
    default_tenure = 12
    default_charges = 65.0
    default_phone = "No"
    default_multiple = "No"
    default_internet = "DSL"
    default_security = "No"
    default_backup = "No"
    default_protection = "No"
    default_support = "No"
    default_tv = "No"
    default_movies = "No"
    default_contract = "Month-to-month"
    default_paperless = "No"
    default_payment = "Electronic check"

st.subheader("Customer Profile Information")

with st.form("customer_form"):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Female", "Male"], index=0 if default_gender == "Female" else 1)
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], index=0 if default_senior == "No" else 1)
        partner = st.selectbox("Partner", ["No", "Yes"], index=0 if default_partner == "No" else 1)
        dependents = st.selectbox("Dependents", ["No", "Yes"], index=0 if default_dependents == "No" else 1)
        
    with col2:
        st.markdown("**Account**")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=default_tenure)
        monthly_charges = st.number_input("Monthly Charges", min_value=18.0, max_value=120.0, value=default_charges)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"], index=0 if default_phone == "No" else 1)
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(default_multiple))
        
    with col3:
        st.markdown("**Internet & Security**")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(default_internet))
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_security))
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_backup))
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_protection))
        
    with col4:
        st.markdown("**Services & Payment**")
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_support))
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_tv))
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_movies))
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(default_contract))
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], index=0 if default_paperless == "No" else 1)
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", 
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(default_payment))
    
    threshold = st.slider("Threshold", 0.0, 1.0, 0.5)
    
    submitted = st.form_submit_button("Starting Predict Churn", type="primary", use_container_width=True)
    
if submitted:
    customer_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges
    }
    
    with st.spinner("Processing customer data..."):
        processed_data = preprocess_telco_data(customer_data)
        
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0]
        churn_prob = probability[1]
        
        prediction_label = "High Risk" if churn_prob >= threshold else "Low Risk"
        
        if churn_prob >= threshold:
            st.error(f"**Prediction:** {prediction_label}")
        else:
            st.success(f"**Prediction:** {prediction_label}")

        st.info(f"""
        **Likelihood to Leave**: **{churn_prob:.1%}**  
        **Likelihood to Stay**: **{1 - churn_prob:.1%}**
        """)

            
            