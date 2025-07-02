```markdown
# 🖱️ Advertising Click Predictor

A full **end-to-end machine learning pipeline** to predict whether a user will click on an online ad, built with **Logistic Regression**, clean modular code, and industry-standard structure.

---

## 📂 Project Structure

```

advertising-click-predictor/
├── config/                 # Configuration YAML for paths and hyperparameters
├── data/
│   ├── raw/                # Original dataset (advertising.csv)
│   ├── processed/          # Train, val, test CSVs
│   ├── interim/            # (Optional) cleaned data
│   └── features/           # (Optional) engineered features
├── models/
│   └── logistic\_model.pkl  # Trained model artifact
├── outputs/
│   ├── confusion\_matrix.png  # Confusion matrix plot
│   ├── roc\_curve.png         # ROC curve plot
│   └── classification\_report.txt # Detailed classification metrics
├── src/
│   ├── data/               # Preprocessing scripts
│   ├── features/           # Feature engineering scripts
│   ├── models/             # Training & evaluation scripts
│   └── utils/              # Common helper utilities
├── run.py                  # 🔁 Full pipeline runner
├── requirements.txt
└── README.md

````

---

## 📈 Dataset

- Source: `advertising.csv`  
- Features:  
  - `Daily Time Spent on Site`, `Age`, `Area Income`, `Daily Internet Usage`, `Ad Topic Line`, `City`, `Male`, `Country`, `Timestamp`
- Target:
  - `Clicked on Ad` (0 = No, 1 = Yes)

---

## ⚙️ How to Run

**Step 1️⃣: Preprocess**

```bash
python src/data/preprocess.py
````

**Step 2️⃣: Build Features**

```bash
python src/features/build_features.py
```

**Step 3️⃣: Train & Validate**

```bash
python src/models/train_model.py
```

**Step 4️⃣: Evaluate on Test**

```bash
python src/models/evaluate_test.py
```

**Or Run Entire Pipeline:**

```bash
python run.py
```

---

## ✅ Final Results

| Metric        | Value    |
| ------------- | -------- |
| Test Accuracy | **0.97** |
| Precision (0) | 0.96     |
| Recall (0)    | 0.99     |
| F1-score (0)  | 0.98     |
| Precision (1) | 0.99     |
| Recall (1)    | 0.96     |
| F1-score (1)  | 0.97     |

---

## 📊 Outputs


## 🏆 Author
MD TARIKUL ISLAM UTHSO
````