from sklearn.metrics import classification_report
from preprocessing import load_and_preprocess
import joblib

def evaluate_model(data_path, model_path):
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)
