import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import joblib

def preprocess(text):
    """
    Applies a series of text preprocessing steps.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    text = str(text).lower()
    text = re.sub(r"http\\S+", "", text)         
    text = re.sub(r"@\\w+", "", text)                
    text = re.sub(r"[^a-zA-Z0-9\\s!?']", "", text)   
    text = re.sub(r"\\s+", " ", text).strip()        
    return text

def create_and_save_model(data_path, model_filename="sentiment_pipeline.pkl"):
    """
    Loads data, preprocesses it, trains a sentiment classification pipeline,
    and saves the trained pipeline to a .pkl file.

    Args:
        data_path (str): Path to the raw CSV data file.
        model_filename (str): Name of the file to save the trained model.
    """
    df = pd.read_csv(data_path, encoding='latin-1', header=None)
    df.columns = ['target', 'id', 'date', 'query', 'user', 'text']

    label_map = {0: 0, 4: 1}
    df["label"] = df["target"].map(label_map)

    df = df[["text", "label"]]

    df["text"] = df["text"].apply(preprocess)

    df_sampled, _ = train_test_split(
        df,
        stratify=df["label"],
        train_size=1000000,  
        random_state=42
    )

    X = df_sampled["text"]
    y = df_sampled["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=7000, ngram_range=(1, 2), min_df=3, max_df=0.9)),
        ("clf", LogisticRegression(C=0.5447697631871001, penalty="l2", max_iter=1000))
    ])

    print("Training the model...")
    pipeline.fit(X_train, y_train)
    print("Model training complete.")

    joblib.dump(pipeline, model_filename)
    print(f"Model saved as {model_filename}")

if __name__ == "__main__":
    data_file_path = 'training.1600000.processed.noemoticon.csv' 
    
create_and_save_model("data/sentiment_data.csv", "models/SentimentCLF.pkl")
