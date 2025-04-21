import joblib
import pandas as pd

def predict(input_data_path, model_path):
    model = joblib.load(model_path)
    data = pd.read_csv(input_data_path)
    return model.predict(data)
