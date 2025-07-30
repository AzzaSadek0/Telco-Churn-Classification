# Telco Customer Churn Prediction System

A comprehensive machine learning-powered web application that predicts customer churn using both sentiment analysis from customer feedback and structured telco customer profile data. Built with Streamlit for an interactive user experience.

## Features

- 🤖 **Sentiment Analysis Model**: Logistic Regression with TF-IDF for text-based churn prediction
- 🗳️ **Voting Classifier**: Ensemble model (GaussianNB, AdaBoost, SVM, LogisticRegression) for telco customer profile-based churn prediction
- 📊 **Interactive Dashboard**: Real-time model predictions and visualizations
- 📝 **Multi-page Interface**: Streamlit-based web application with navigation
- 🔍 **Model Comparison**: Side-by-side performance analysis
- ⚠️ **Educational Purpose**: Clear disclaimer for demonstration use

## Project Structure

```
telco-churn-prediction/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── app.py                       # Main Streamlit application entry point
├── pages/                       # Streamlit multi-page app components
│   ├── Dashboard.py             # Main dashboard with navigation
│   ├── SentimentAnalysis.py     # Text-based churn prediction interface
│   └── TelcoChurnPrediction.py  # Profile-based churn prediction interface
├── src/                         # Core machine learning logic and utilities
│   ├── model.py                 # Model loader and prediction interface
│   ├── ChurnVotingCLF.py        # Training script for ensemble model
│   ├── ChurnSentimentCLF.py     # Training script for sentiment model
│   └── preprocessing.py         # Data cleaning and transformation functions
├── models/                      # Saved model artifacts
│   ├── VotingCLF.pkl            # Trained VotingClassifier
│   ├── SentimentCLF.pkl         # Trained Logistic Regression for sentiment
│   └── Scaler.pkl               # Fitted StandardScaler
├── data/                        # Input datasets
├── notebooks/                   # Jupyter notebooks for experimentation
│   ├── ChurnVotingCLF.ipynb     # Ensemble model development
│   └── ChurnSentimentCLF.ipynb  # Sentiment model development
└── mlruns/                      # MLflow experiment tracking logs
```

## Usage

### Quick Start

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

3. **Navigate through the application:**
   - **Dashboard**: Overview and model comparison
   - **Sentiment Analysis**: Text-based churn prediction
   - **Telco Churn Prediction**: Profile-based churn prediction

### MLflow Experiment Tracking

To view the experiment tracking and model performance metrics:

1. **Start MLflow UI:**
    ```bash
    mlflow ui
    ```

2. **Access the MLflow dashboard:**
   - Open your browser and go to: `http://localhost:5000`
   - Browse through different experiments:
     - `pure-ml-models`: Individual model performance
     - `tuned-pure-ml-models`: Hyperparameter tuned models
     - `tuned-pure-ml-models-smoteenn`: Models with SMOTEENN balancing
     - `Voting-Classifier-model`: Final ensemble model

3. **Compare model metrics:**
   - View training/test accuracy, F1 scores
   - Compare hyperparameters across runs
   - Download model artifacts and reports

## Model Information

### Sentiment Analysis Model
- **Algorithm**: Logistic Regression with TF-IDF Vectorization
- **Purpose**: Predicts churn based on customer feedback text
- **Features**: 7,000 TF-IDF features with n-grams (1,2)
- **Performance**: 
  - Optimized hyperparameters: C=0.545, penalty='l2'
  - Training on 1M samples for efficiency

### VotingClassifier (Telco Churn)
- **Algorithm**: Ensemble of GaussianNB, AdaBoost, SVM, LogisticRegression
- **Purpose**: Predicts churn using telco customer profile data
- **Features**: 23 engineered features including demographics and service usage
- **Performance**:
  - **Train Accuracy**: 95.98%
  - **Test Accuracy**: 93.91%
  - **F1 Score**: Weighted average across classes
- **Optimization**: Hyperparameters tuned using Optuna

### Key Technical Features:
- **Preprocessing Pipeline**: Feature engineering, standardization, encoding
- **Model Persistence**: Saved models using joblib for production use
- **Real-time Prediction**: Interactive web interface for instant predictions
- **Visualization**: Confusion matrices, probability distributions, feature importance
- **Experiment Tracking**: MLflow integration for model versioning and comparison
- **Hyperparameter Optimization**: Optuna-based automated tuning
- **Model Interpretability**: LIME explanations for individual predictions

## Technical Stack
| Category               | Tools / Libraries                                 |
| ---------------------- | ------------------------------------------------- |
| ML Framework           | `scikit-learn`, `xgboost`, `lightgbm`, `catboost` |
| Optimization           | `Optuna`                                          |
| Web App                | `Streamlit`                                       |
| Tracking               | `MLflow`                                          |
| Data Processing        | `pandas`, `numpy`, `SMOTEENN`, `TF-IDF`, `Scaler` |
| Visualization          | `plotly`, `matplotlib`, `seaborn`                 |
| Model Interpretability | `LIME`                                            |
| Persistence            | `joblib`                                          |

## Important Disclaimer

**This application is designed for educational and demonstration purposes only.**

**Predictions are not 100% accurate and should not be used for critical business decisions without proper validation and additional analysis.**