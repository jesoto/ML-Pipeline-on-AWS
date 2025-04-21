
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42)
