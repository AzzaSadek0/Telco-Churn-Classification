import streamlit as st
import joblib
import sys
import os
from lime.lime_text import LimeTextExplainer
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import preprocess_text 
from matplotlib.patches import Patch

@st.cache_resource
def load_model():
    return joblib.load('models/SentimentCLF.pkl')

model = load_model()

st.title("📝 LogReg - Sentiment Analysis")
st.markdown("---")
st.markdown("""
This model analyzes customer feedback text to predict the likelihood of churn.
It uses sentiment analysis and text features to make predictions.
""")

st.sidebar.header("Test Cases")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("Low Risk Test", type="secondary", use_container_width=True):
        st.session_state.test_text = "I love this service! The customer support is amazing and everything works perfectly. Very satisfied with my experience and highly recommend it."

with col2:
    if st.button("High Risk Test", type="secondary", use_container_width=True):
        st.session_state.test_text = "This service is terrible! I'm constantly having issues, the support is awful, and I'm thinking about canceling my subscription immediately. Very disappointed."

st.sidebar.markdown("---")
st.sidebar.header("Model Information")

with st.sidebar.expander("How Logistic Regression Works", expanded=False):
    st.markdown("""
    **Statistical Model**: Uses probability to make predictions
    
    **Text Processing Pipeline**:
    - 📝 **Tokenization**: Breaks text into words
    - 🧹 **Cleaning**: Removes stopwords, punctuation
    - 🔢 **Vectorization**: Converts text to numbers (TF-IDF)
    - ⚖️ **Classification**: Applies logistic function
    
    **Key Advantages**:
    - ⚡ Fast and efficient
    - 📊 Interpretable results
    - 🎯 Good for text classification
    - 🎲 Provides probability scores
    """)

with st.sidebar.expander("Model Performance", expanded=False):
    st.markdown("""
    **Accuracy**: ~80% on test data
    
    **Strengths**:
    - ✅ Fast text processing
    - ✅ Good sentiment detection
    - ✅ Handles various text lengths
    - ✅ Provides confidence scores
    
    **Best Used For**:
    - 📧 Email sentiment analysis
    - 💬 Chat/feedback classification
    - 🔍 Quick sentiment screening
    - 📊 Bulk text processing
    """)

with st.sidebar.expander("LIME Explanations", expanded=False):
    st.markdown("""
    **What is LIME?**
    - 🔍 **Local**: Explains individual predictions
    - 🎯 **Interpretable**: Shows word importance
    - 📊 **Visual**: Color-coded contributions
    
    **How to Read**:
    - 🔴 **Red bars**: Increase churn risk
    - 🟢 **Green bars**: Decrease churn risk
    - 📏 **Bar length**: Strength of impact
    
    **Use Cases**:
    - Understanding model decisions
    - Identifying key sentiment drivers
    - Quality control for predictions
    """)

st.sidebar.markdown("---")

st.subheader("Enter Customer Feedback")
user_input = st.text_area(
    "Customer Feedback", 
    value=st.session_state.get('test_text', ''),
    placeholder="Enter customer feedback or complaint text here...",
    help="Paste the customer's message, complaint, or feedback text"
)

threshold = st.slider(
    "Set Probability Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Adjust the threshold for predicting churn. Default is 0.5"
)

show_explanation = st.checkbox("Show LIME Explanation", help="Generate explanation for model prediction")

predict_button = st.button("Start Analyzing", type="primary", use_container_width=True)

if predict_button:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some feedback text.")
    else:
        with st.spinner("Analyzing feedback..."):
            try:
                processed_text = preprocess_text(user_input)
                proba = model.predict_proba([processed_text])[0]
                churn_prob = proba[0]

                prediction_label = "High Risk" if churn_prob >= threshold else "Low Risk"

                if churn_prob >= threshold:
                    st.error(f"**Prediction:** {prediction_label}")
                else:
                    st.success(f"**Prediction:** {prediction_label}")
                st.info(f"**Probability:** {churn_prob:.3f}")

                if show_explanation:
                    st.subheader("Explanation (LIME)")

                    explainer = LimeTextExplainer(class_names=['High Risk', 'Low Risk'])

                    def predict_fn(texts):
                        return model.predict_proba([preprocess_text(text) for text in texts])

                    explanation = explainer.explain_instance(
                        user_input,
                        predict_fn,
                        num_features=10,
                        labels=[0]
                    )

                    exp_list = explanation.as_list(label=0)
                    words, weights = zip(*exp_list)
                    sorted_items = sorted(zip(words, weights), key=lambda x: x[1])
                    words_sorted, weights_sorted = zip(*sorted_items)

                    colors = ['#d62728' if w > 0 else '#2ca02c' for w in weights_sorted]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.barh(words_sorted, weights_sorted, color=colors, edgecolor='black')

                    ax.set_title("Word Contribution to Prediction (LIME Explanation)", fontsize=16, pad=15)
                    ax.set_xlabel("Impact on Prediction", fontsize=12)
                    ax.axvline(0, color='gray', linewidth=1.2, linestyle='--')
                    ax.grid(True, axis='x', linestyle=':', linewidth=0.7, alpha=0.7)

                    ax.tick_params(axis='both', labelsize=11)

                    legend_elements = [
                        Patch(facecolor='#d62728', edgecolor='black', label='Increases Risk'),
                        Patch(facecolor='#2ca02c', edgecolor='black', label='Decreases Risk')
                    ]
                    ax.legend(handles=legend_elements, loc='lower right', frameon=True)

                    st.pyplot(fig)
                    

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
