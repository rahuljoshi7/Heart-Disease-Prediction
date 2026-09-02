from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

# Load Heart Disease dataset from UCI
heart_disease = fetch_ucirepo(id=45)

# Get features and target
X = heart_disease.data.features
y = heart_disease.data.targets

# Combine them
data = pd.concat([X, y], axis=1)

# Convert target into binary classification
# 0 = Low/No Risk
# 1 = High Risk
data["num"] = data["num"].apply(lambda x: 0 if x == 0 else 1)

# Save dataset
os.makedirs("data", exist_ok=True)

data.to_csv("data/heart_disease.csv", index=False)

print("Dataset downloaded successfully!")
print("Dataset shape:", data.shape)
print(data.head())