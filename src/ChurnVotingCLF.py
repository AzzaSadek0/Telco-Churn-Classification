import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


def preprocess_telco_data(data):
    """
    Preprocesses telco customer data for churn prediction.
    
    Args:
        data (dict): Dictionary containing customer data
        
    Returns:
        pd.DataFrame: Preprocessed data ready for model prediction
    """
    df = pd.DataFrame(data, index=[0])
    
    df["RealTotalCharges"] = df["MonthlyCharges"] * df["tenure"]
    
    cols_to_replace = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    for col in cols_to_replace:
        if col in df.columns:
            df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})
    
    try:
        scaler = joblib.load('models/Scaler.pkl')
        numerical_cols = ['tenure', 'MonthlyCharges', 'RealTotalCharges']
        df[numerical_cols] = scaler.transform(df[numerical_cols])
    except FileNotFoundError:
        print("Warning: Scaler not found. Using StandardScaler with default parameters.")
        scaler = StandardScaler()
        numerical_cols = ['tenure', 'MonthlyCharges', 'RealTotalCharges']
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    if 'SeniorCitizen' not in df.columns:
        df['SeniorCitizen'] = 0
    
    binary_mapping = {
        'gender': {'Female': 0, 'Male': 1},
        'SeniorCitizen': {0: 0, 1: 1},  
        'Partner': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1},
        'PhoneService': {'No': 0, 'Yes': 1},
        'MultipleLines': {'No': 0, 'Yes': 1},
        'OnlineSecurity': {'No': 0, 'Yes': 1},
        'OnlineBackup': {'No': 0, 'Yes': 1},
        'DeviceProtection': {'No': 0, 'Yes': 1},
        'TechSupport': {'No': 0, 'Yes': 1},
        'StreamingTV': {'No': 0, 'Yes': 1},
        'StreamingMovies': {'No': 0, 'Yes': 1},
        'PaperlessBilling': {'No': 0, 'Yes': 1}
    }
    
    for col in binary_mapping:
        if col in df.columns:
            df[col] = df[col].map(binary_mapping[col])
    
    multi_cols = ['InternetService', 'Contract', 'PaymentMethod']
    existing_multi_cols = [col for col in multi_cols if col in df.columns]
    if existing_multi_cols:
        df = pd.get_dummies(df, columns=existing_multi_cols, drop_first=True)
    
    model_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
        'MonthlyCharges', 'RealTotalCharges', 'InternetService_Fiber optic',
        'InternetService_No', 'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]
    
    df = df.reindex(columns=model_cols, fill_value=0)
    
    return df


def create_voting_classifier():
    """
    Creates and returns the optimized VotingClassifier model.
    
    Returns:
        VotingClassifier: Trained voting classifier model
    """
    best_logistic_model = LogisticRegression(
        C=0.4030592884477322,
        penalty='l2',
        solver='liblinear',
        class_weight='balanced',
        random_state=42
    )
    
    best_nb_model = GaussianNB()
    
    best_ada_model = AdaBoostClassifier(
        n_estimators=107,
        learning_rate=0.28587237664368714,
        random_state=42
    )
    
    best_svm_model = SVC(
        C=4.730871681942921,
        kernel='rbf',
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    
    voting_clf = VotingClassifier(
        estimators=[
            ('nb', best_nb_model),
            ('ada', best_ada_model),
            ('svm', best_svm_model),
            ('lr', best_logistic_model)
        ],
        voting='soft'
    )
    
    return voting_clf


def save_model(model, scaler=None):
    """
    Saves the trained model and scaler to pickle files.
    
    Args:
        model: The trained model to save
        scaler: The fitted scaler to save (optional)
    """
    import os
    
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, 'models/VotingCLF.pkl')
    print("Model saved successfully to models/VotingCLF.pkl")
    
    if scaler is not None:
        joblib.dump(scaler, 'models/Scaler.pkl')
        print("Scaler saved successfully to models/Scaler.pkl")


def predict_churn(customer_data):
    """
    Predicts churn probability for a customer.
    
    Args:
        customer_data (dict): Dictionary containing customer features
        
    Returns:
        tuple: (prediction, probability) where prediction is 0/1 and probability is float
    """
    try:
        model = joblib.load('models/VotingCLF.pkl')
    except FileNotFoundError:
        print("Warning: Pre-trained model not found. Creating new model.")
        model = create_voting_classifier()
    
    processed_data = preprocess_telco_data(customer_data)
    
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0]
    
    return prediction, probability