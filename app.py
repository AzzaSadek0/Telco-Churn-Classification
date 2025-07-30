import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Churn Prediction Models",
    page_icon="📊",
    layout="wide"
)

st.title("Churn Prediction Models")
st.markdown("---")


col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📝 **Logistic Regression Model**
    - **Purpose**: Predicts churn based on customer feedback text  
    - **Input**: Customer feedback/complaint text  
    - **Use Case**: Analyze customer sentiment and predict likelihood of churn from their messages  

    **📊 Performance:**  
    - Train Accuracy: **80.21%**  
    - Test Accuracy: **79.91%**  
    """)
    if st.button("📝 Go to Sentiment Analysis", type="primary", use_container_width=True):
        st.switch_page("pages/SentimentAnalysis.py")

with col2:
    st.markdown("""
    ### 🗳️ **Voting Classifier Model (NB - Ada - SVM - LR)**
    - **Purpose**: Predicts churn based on customer profile data  
    - **Input**: Customer demographic and service usage data  
    - **Use Case**: Analyze customer characteristics to predict churn probability  

    **📊 Performance:**  
    - Train Accuracy: **95.98%**  
    - Test Accuracy: **93.91%**  
    """)
    if st.button("🗳️ Go to Telco Churn Prediction", type="primary", use_container_width=True):
        st.switch_page("pages/TelcoChurnPrediction.py")


st.markdown("---")

if st.button("📊 Open Analytics Dashboard", type="secondary", use_container_width=True):
    st.switch_page("pages/Dashboard.py")
