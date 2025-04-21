import joblib
from sklearn.ensemble import RandomForestClassifier
from preprocessing import load_and_preprocess

def train_model(data_path, model_output_path):
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, model_output_path)
