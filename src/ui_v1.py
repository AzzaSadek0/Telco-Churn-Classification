import streamlit as st
import joblib

# Load saved model
model = joblib.load("models/ChurnFeedbackCLF.pkl")

# App title
st.title("Churn Prediction from Customer Feedback")
st.write("Enter a customer message to see if it indicates churn.")

# Text input
user_input = st.text_area("Customer Feedback", placeholder="Type or paste customer feedback here...")

# Predict button
if st.button("Predict Churn"):
    if user_input.strip() == "":
        st.warning("Please enter some feedback text.")
    else:
        prediction = model.predict([user_input])[0]
        probability = model.predict_proba([user_input])[0].max()

        label = "Churn" if prediction == 1 else "Not Churn"
        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {probability:.2f}")
