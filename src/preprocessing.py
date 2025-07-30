import re
import pandas as pd
import joblib

scaler = joblib.load('models/scaler.pkl')

def preprocess_text(text):

    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)            
    text = re.sub(r"@\w+", "", text)                
    text = re.sub(r"[^a-zA-Z0-9\s!?']", "", text)   
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_telco_data(data):

    df = pd.DataFrame(data, index=[0])

    df["RealTotalCharges"] = df["MonthlyCharges"] * df["tenure"]
    
    cols_to_replace = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in cols_to_replace:
        df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})

    numerical_cols = ['tenure', 'MonthlyCharges', 'RealTotalCharges']
    df[numerical_cols] = scaler.transform(df[numerical_cols])

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
        df[col] = df[col].map(binary_mapping[col])

    multi_cols = ['InternetService', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    
    model_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
       'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
       'MonthlyCharges', 'RealTotalCharges', 'InternetService_Fiber optic',
       'InternetService_No', 'Contract_One year', 'Contract_Two year',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']
       
    df = df.reindex(columns=model_cols, fill_value=0)

    return df
