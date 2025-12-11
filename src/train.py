import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = os.path.join("data", "processed")
MODELS_PATH = "models"
MODEL_NAME = "logistic_model.joblib"

# Ensure models directory exists
os.makedirs(MODELS_PATH, exist_ok=True)

def train():
    print("Loading training data...")
    # Load features and target
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "train_features.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "train_target.csv"))
    
    # Flatten y_train (sklearn requires a 1D array for targets)
    y_train = y_train.values.ravel()

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # Optional: Quick check on training accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    print(f"Training Accuracy: {train_acc:.4f}")

    # Save the model
    save_path = os.path.join(MODELS_PATH, MODEL_NAME)
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()