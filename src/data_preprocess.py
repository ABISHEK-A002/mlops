import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
RAW_DATA_PATH = os.path.join("data", "raw")
PROCESSED_DATA_PATH = os.path.join("data", "processed")
RAW_FILENAME = "iris.csv"

# Ensure directories exist
os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

def load_data():
    """Simulates loading raw data. If file doesn't exist, download it."""
    file_path = os.path.join(RAW_DATA_PATH, RAW_FILENAME)
    
    if not os.path.exists(file_path):
        print(f"Raw data not found. Downloading to {file_path}...")
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        df.to_csv(file_path, index=False)
    else:
        print(f"Loading raw data from {file_path}...")
        df = pd.read_csv(file_path)
    
    return df

def process_data(df):
    """Splits data into train and test sets."""
    print("Splitting data...")
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 80% Train, 20% Test
    return train_test_split(X, y, test_size=0.2, random_state=42)

def save_data(X_train, X_test, y_train, y_test):
    """Saves processed data to disk."""
    print(f"Saving processed files to {PROCESSED_DATA_PATH}...")
    X_train.to_csv(os.path.join(PROCESSED_DATA_PATH, "train_features.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DATA_PATH, "test_features.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DATA_PATH, "train_target.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DATA_PATH, "test_target.csv"), index=False)
    print("Preprocessing complete!")

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = process_data(df)
    save_data(X_train, X_test, y_train, y_test)