from src.utils import ensure_data_path, encode_labels
import pandas as pd

def load_data(path):
    path = ensure_data_path(path)
    df = pd.read_csv(path)
    df['hired'] = encode_labels(df['hired'])
    X = df.drop('hired', axis=1)
    y = df['hired']
    return X, y
