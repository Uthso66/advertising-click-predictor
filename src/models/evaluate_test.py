import pandas as pd
import yaml
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def evaluate_test(config):
    features_dir = config['data']['features_dir']
    processed_dir = config['data']['processed_dir']

    X_test = pd.read_csv(f"{features_dir}/X_test.csv")
    y_test = pd.read_csv(f"{processed_dir}/y_test.csv").values.ravel()

    model = joblib.load(config['model']['save_path'])

    y_pred = model.predict(X_test)
    print("🧪 Test Classification Report:\n")
    report = classification_report(y_test, y_pred)
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.savefig(config['evaluation']['confusion_matrix_plot'])
    print(f"✅ Confusion matrix saved to {config['evaluation']['confusion_matrix_plot']}")

    # ROC Curve
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(config['evaluation']['roc_curve_plot'])
    print(f"✅ ROC curve saved to {config['evaluation']['roc_curve_plot']}")

    # Save text report too
    with open(config['evaluation']['classification_report_txt'], 'w') as f:
        f.write(report)
    print(f"📄 Classification report saved to {config['evaluation']['classification_report_txt']}")


def main():
    config = load_config()
    evaluate_test(config)

if __name__ == '__main__':
    main()
