"""
LoanSentry - Training Script 1: Data Preprocessing
===================================================
Covers:
- Dataset loading from Kaggle
- Exploratory Data Analysis
- Missing value handling
- Feature engineering
- Encoding
- Train-test split
- SMOTE for class imbalance
- Scaling and export
"""

import os
import pickle
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG - update these paths before running
# ─────────────────────────────────────────────
DATASET_PATH = "data/accepted_2007_to_2018Q4.csv"   # path to downloaded CSV
SAVE_PATH    = "output/"                              # where to save outputs
NROWS        = 500000                                 # rows to load

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs("plots/", exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH, nrows=NROWS, low_memory=False)
print(f"Dataset shape: {df.shape}")

# ─────────────────────────────────────────────
# 2. TARGET VARIABLE
# ─────────────────────────────────────────────
print("\nLoan Status Values:")
print(df["loan_status"].value_counts())

df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])]
df["TARGET"] = (df["loan_status"] == "Charged Off").astype(int)
df = df.drop(columns=["loan_status"])

print(f"\nDataset shape after filtering: {df.shape}")
print(f"Default rate: {df['TARGET'].mean()*100:.2f}%")

# ─────────────────────────────────────────────
# 3. EDA PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["TARGET"].value_counts().plot(
    kind="bar", ax=axes[0], color=["steelblue", "tomato"])
axes[0].set_title("Target Distribution (Count)")
axes[0].set_xlabel("Target (0 = Fully Paid, 1 = Default)")
axes[0].set_ylabel("Count")
axes[0].set_xticklabels(["Fully Paid", "Default"], rotation=0)

df["TARGET"].value_counts().plot(
    kind="pie", ax=axes[1],
    labels=["Fully Paid", "Default"],
    autopct="%1.1f%%",
    colors=["steelblue", "tomato"])
axes[1].set_title("Target Distribution (%)")
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig("plots/target_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plots/target_distribution.png")

# ─────────────────────────────────────────────
# 4. FEATURE SELECTION
# ─────────────────────────────────────────────
selected_features = [
    "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "emp_length", "home_ownership",
    "annual_inc", "verification_status", "purpose", "dti",
    "delinq_2yrs", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "initial_list_status",
    "application_type", "mort_acc", "pub_rec_bankruptcies",
    "TARGET"
]
df = df[selected_features]
print(f"\nShape after feature selection: {df.shape}")

# ─────────────────────────────────────────────
# 5. MISSING VALUES
# ─────────────────────────────────────────────
missing     = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df  = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_pct
}).sort_values("Missing %", ascending=False)
missing_df = missing_df[missing_df["Missing Count"] > 0]
print(f"\nColumns with missing values: {len(missing_df)}")
print(missing_df)

plt.figure(figsize=(10, 6))
sns.barplot(x=missing_df["Missing %"], y=missing_df.index, palette="Reds_r")
plt.title("Missing Value % per Column")
plt.xlabel("Missing %")
plt.tight_layout()
plt.savefig("plots/missing_values.png", dpi=150, bbox_inches="tight")
plt.close()

num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in num_cols if c != "TARGET"]

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print(f"Missing values remaining: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 6. CLEAN STRING COLUMNS
# ─────────────────────────────────────────────
df["term"] = df["term"].str.strip().str.replace(" months", "").astype(int)

if df["int_rate"].dtype == object:
    df["int_rate"] = df["int_rate"].str.replace("%", "").astype(float)

if df["revol_util"].dtype == object:
    df["revol_util"] = df["revol_util"].str.replace("%", "").astype(float)

emp_map = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
    "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
    "8 years": 8, "9 years": 9, "10+ years": 10
}
df["emp_length"] = df["emp_length"].map(emp_map).fillna(0)
print("String columns cleaned")

# ─────────────────────────────────────────────
# 7. EDA - DISTRIBUTIONS AND CORRELATIONS
# ─────────────────────────────────────────────
key_features = ["loan_amnt", "int_rate", "annual_inc",
                "dti", "fico_range_low", "installment"]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(key_features):
    axes[i].hist(df[col], bins=50, color="steelblue",
                 edgecolor="white", alpha=0.8)
    axes[i].set_title(col)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")
plt.suptitle("Key Feature Distributions", fontsize=14)
plt.tight_layout()
plt.savefig("plots/feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 5))
grade_default = df.groupby("grade")["TARGET"].mean().sort_index() * 100
sns.barplot(x=grade_default.index, y=grade_default.values, palette="RdYlGn_r")
plt.title("Default Rate by Loan Grade")
plt.xlabel("Grade")
plt.ylabel("Default Rate %")
plt.tight_layout()
plt.savefig("plots/default_by_grade.png", dpi=150, bbox_inches="tight")
plt.close()

corr_cols = ["loan_amnt", "int_rate", "annual_inc", "dti",
             "fico_range_low", "installment", "revol_bal", "TARGET"]
plt.figure(figsize=(10, 8))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("EDA plots saved to plots/")

# ─────────────────────────────────────────────
# 8. FEATURE ENGINEERING
# ─────────────────────────────────────────────
df["LOAN_TO_INCOME"]        = df["loan_amnt"] / (df["annual_inc"] + 1)
df["FICO_AVG"]              = (df["fico_range_low"] + df["fico_range_high"]) / 2
df["INSTALLMENT_TO_INCOME"] = df["installment"] / (df["annual_inc"] / 12 + 1)
df["FUNDED_RATIO"]          = df["funded_amnt"] / (df["loan_amnt"] + 1)
df = df.drop(columns=["fico_range_low", "fico_range_high"])

print(f"Feature engineering complete. Shape: {df.shape}")

# ─────────────────────────────────────────────
# 9. ENCODING
# ─────────────────────────────────────────────
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
print(f"Encoding columns: {cat_cols}")

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))
print("Encoding complete")

# ─────────────────────────────────────────────
# 10. TRAIN-TEST SPLIT
# ─────────────────────────────────────────────
X = df.drop(columns=["TARGET"])
y = df["TARGET"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set:     {X_test.shape}")

# ─────────────────────────────────────────────
# 11. SMOTE
# ─────────────────────────────────────────────
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {X_train.shape[0]} samples")
print(f"After SMOTE:  {X_train_sm.shape[0]} samples")

# ─────────────────────────────────────────────
# 12. SCALING
# ─────────────────────────────────────────────
scaler          = StandardScaler()
X_train_scaled  = scaler.fit_transform(X_train_sm)
X_test_scaled   = scaler.transform(X_test)
print("Scaling complete")

# ─────────────────────────────────────────────
# 13. SAVE OUTPUTS
# ─────────────────────────────────────────────
np.save(SAVE_PATH + "X_train.npy", X_train_scaled)
np.save(SAVE_PATH + "X_test.npy",  X_test_scaled)
np.save(SAVE_PATH + "y_train.npy", y_train_sm)
np.save(SAVE_PATH + "y_test.npy",  y_test.values)

with open(SAVE_PATH + "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

feature_names = X.columns.tolist()
with open(SAVE_PATH + "feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

for plot in ["target_distribution.png", "missing_values.png",
             "feature_distributions.png", "correlation_heatmap.png",
             "default_by_grade.png"]:
    src = f"plots/{plot}"
    if os.path.exists(src):
        shutil.copy(src, SAVE_PATH + plot)

print(f"\nAll outputs saved to {SAVE_PATH}")
print("Files saved:")
for f in os.listdir(SAVE_PATH):
    print(f"  - {f}")