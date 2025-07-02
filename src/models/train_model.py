import pandas as pd
import yaml
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_model(X_train, y_train, X_val, y_val, config):
    model_params = config['model']['hyperparameters']
    model = LogisticRegression(
        solver=model_params['solver'],
        penalty=model_params['penalty'],
        C=model_params['C']
    )

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_val_pred)
    print(f"✅ Validation Accuracy: {acc:.4f}")
    print("📊 Classification Report:\n", classification_report(y_val, y_val_pred))

    joblib.dump(model, config['model']['save_path'])
    print(f"💾 Model saved to: {config['model']['save_path']}")

def main():
    config = load_config()
    features_dir = config['data']['features_dir']
    processed_dir = config['data']['processed_dir']

    X_train = pd.read_csv(f"{features_dir}/X_train.csv")
    X_val = pd.read_csv(f"{features_dir}/X_val.csv")
    y_train = pd.read_csv(f"{processed_dir}/y_train.csv").values.ravel()
    y_val = pd.read_csv(f"{processed_dir}/y_val.csv").values.ravel()

    train_model(X_train, y_train, X_val, y_val, config)

if __name__ == "__main__":
    main()

