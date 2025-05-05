import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

def ensure_data_path(path):
    """Ensure the data file exists."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return path

def encode_labels(column):
    """Encode categorical labels (e.g., 'Yes'/'No') into 0 and 1."""
    le = LabelEncoder()
    return le.fit_transform(column)

def print_model_results(name, y_test, y_pred):
    """Standardized model evaluation output."""
    print(f"\n{name} - Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
