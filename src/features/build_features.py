import os
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

# Load config.yaml
def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# Updated build_feature function  
def build_feature(X, scaler=None):  
    if 'Male' in X.columns:  
        X['Male'] = X['Male'].astype(int)  
    
    numeric_cols = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']  
    
    if scaler is None:  
        # Train mode: fit new scaler  
        scaler = StandardScaler()  
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])  
    else:  
        # Val/test mode: use existing scaler  
        X[numeric_cols] = scaler.transform(X[numeric_cols])  
    
    return X, scaler  

def main():
    config = load_config()
    processed_dir = config['data']['processed_dir']
    features_dir = config['data']['features_dir']

    os.makedirs(features_dir, exist_ok=True)

    # Processed splits load
    X_train = pd.read_csv(f"{processed_dir}/X_train.csv")
    X_val = pd.read_csv(f"{processed_dir}/X_val.csv")
    X_test = pd.read_csv(f"{processed_dir}/X_test.csv")

    # Apply feature engineering
    # Process train (get fitted scaler)  
    X_train, scaler = build_feature(X_train)  

    # Process val/test with SAME scaler  
    X_val, _ = build_feature(X_val, scaler)  
    X_test, _ = build_feature(X_test, scaler)  

    # Save feature sets
    X_train.to_csv(f"{features_dir}/X_train.csv", index=False)
    X_val.to_csv(f"{features_dir}/X_val.csv", index=False)
    X_test.to_csv(f"{features_dir}/X_test.csv", index=False)

    print("✅ Feature engineering complete and saved.")

if __name__ == "__main__":
    main()
