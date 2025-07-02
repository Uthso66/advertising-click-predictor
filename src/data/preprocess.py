import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Loading config.yaml
def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# Loading raw csv
def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.drop(columns=['Ad Topic Line', 'City', 'Country', 'Timestamp'])
    return df

def split_data(df, test_size, val_size, random_state):
    X = df.drop('Clicked on Ad', axis=1)
    y = df['Clicked on Ad']

    X_temp, X_test, y_temp, y_test, = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )    

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def save_data(X_train, X_val, X_test, y_train, y_val, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_val.to_csv(f"{output_dir}/X_val.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)

    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_val.to_csv(f"{output_dir}/y_val.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

def main():
    config = load_config()
    raw_path = config['data']['raw_path']
    processed_dir = config['data']['processed_dir']
    test_size = config['data']['test_size']
    val_size = config['data']['val_size']
    random_state = config['data']['random_state']

    df = load_data(raw_path)
    df = clean_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, test_size, val_size, random_state
    )
    save_data(X_train, X_val, X_test, y_train, y_val, y_test, processed_dir)
    print('✅ Advertising data preprocessing complete and saved.')

if __name__ == '__main__':
    main()
