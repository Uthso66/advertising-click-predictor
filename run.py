import os

print("\n🚀 Starting Advertising Click Prediction Pipeline...\n")

# 1️⃣ Data Preprocessing
os.system("python src/data/preprocess.py")

# 2️⃣ Feature Engineering
os.system("python src/features/build_features.py")

# 3️⃣ Train & Validate
os.system("python src/models/train_model.py")

# 4️⃣ Evaluate on Test
os.system("python src/models/evaluate_test.py")

print("\n ✅ Pipeline complete. All steps executed successfully.\n")
