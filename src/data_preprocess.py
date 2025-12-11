import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Define paths
RAW_PATH = os.path.join('data', 'raw')
PROCESSED_PATH = os.path.join('data', 'processed')

os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)

def preprocess():
    print("Loading data...")
    # 1. Simulate Data Ingestion (Save to Raw)
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    raw_file = os.path.join(RAW_PATH, 'iris_raw.csv')
    df.to_csv(raw_file, index=False)
    print(f"Raw data saved to {raw_file}")

    # 2. Processing (Split Data)
    print("Splitting data...")
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Save Processed Data
    X_train.to_csv(os.path.join(PROCESSED_PATH, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(PROCESSED_PATH, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(PROCESSED_PATH, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(PROCESSED_PATH, 'y_test.csv'), index=False)
    
    print(f"Processed data saved to {PROCESSED_PATH}")

if __name__ == "__main__":
    preprocess()