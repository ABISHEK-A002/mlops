import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

PROCESSED_PATH = os.path.join('data', 'processed')
MODELS_PATH = 'models'

os.makedirs(MODELS_PATH, exist_ok=True)

def train():
    print("Loading training data...")
    X_train = pd.read_csv(os.path.join(PROCESSED_PATH, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_PATH, 'y_train.csv')).values.ravel()

    # Initialize and train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(MODELS_PATH, 'rf_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()