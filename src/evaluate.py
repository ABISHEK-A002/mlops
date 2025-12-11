import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

PROCESSED_PATH = os.path.join('data', 'processed')
MODELS_PATH = 'models'

def evaluate():
    print("Loading test data and model...")
    X_test = pd.read_csv(os.path.join(PROCESSED_PATH, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(PROCESSED_PATH, 'y_test.csv')).values.ravel()

    model_path = os.path.join(MODELS_PATH, 'rf_model.joblib')
    model = joblib.load(model_path)

    # Predictions
    print("Evaluating...")
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print("-" * 30)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("-" * 30)
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    evaluate()