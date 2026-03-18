"""
LoanSentry - Training Script 2: Traditional ML Models
=====================================================
Covers:
- Logistic Regression, Random Forest, XGBoost
- Evaluation and comparison
- SHAP explainability
- Saving models
"""

import os
import json
import pickle
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb_lib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, roc_auc_score,
                              roc_curve)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LOAD_PATH = "output/"    # where 01_data_preprocessing.py saved its outputs
SAVE_PATH = "output/"
PLOT_PATH = "plots/"

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(PLOT_PATH, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
X_train = np.load(LOAD_PATH + "X_train.npy")
X_test  = np.load(LOAD_PATH + "X_test.npy")
y_train = np.load(LOAD_PATH + "y_train.npy")
y_test  = np.load(LOAD_PATH + "y_test.npy")

with open(LOAD_PATH + "feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"Features:      {len(feature_names)}")

# ─────────────────────────────────────────────
# 2. EVALUATION HELPER
# ─────────────────────────────────────────────
def evaluate_model(name, model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*40}")
    print(f"Model: {name} (threshold={threshold})")
    print(f"{'='*40}")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc:.4f}")
    print(classification_report(y_test, y_pred))

    return {
        "name": name, "model": model,
        "y_pred": y_pred, "y_prob": y_prob,
        "accuracy": acc, "f1": f1, "roc_auc": roc,
        "threshold": threshold
    }

# ─────────────────────────────────────────────
# 3. LOGISTIC REGRESSION
# ─────────────────────────────────────────────
print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr.fit(X_train, y_train)
lr_results = evaluate_model("Logistic Regression", lr, X_test, y_test)

# ─────────────────────────────────────────────
# 4. RANDOM FOREST
# ─────────────────────────────────────────────
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_results = evaluate_model("Random Forest", rf, X_test, y_test)

# ─────────────────────────────────────────────
# 5. XGBOOST
# ─────────────────────────────────────────────
print("Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42, n_jobs=-1
)
xgb.fit(X_train, y_train)

XGB_THRESHOLD = 0.3
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_prob_xgb >= XGB_THRESHOLD).astype(int)

acc = accuracy_score(y_test, y_pred_xgb)
f1  = f1_score(y_test, y_pred_xgb)
roc = roc_auc_score(y_test, y_prob_xgb)

print(f"\n{'='*40}")
print(f"Model: XGBoost (threshold={XGB_THRESHOLD})")
print(f"{'='*40}")
print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc:.4f}")
print(classification_report(y_test, y_pred_xgb))

xgb_results = {
    "name": "XGBoost", "model": xgb,
    "y_pred": y_pred_xgb, "y_prob": y_prob_xgb,
    "accuracy": acc, "f1": f1, "roc_auc": roc,
    "threshold": XGB_THRESHOLD
}

# ─────────────────────────────────────────────
# 6. COMPARISON TABLE
# ─────────────────────────────────────────────
results = [lr_results, rf_results, xgb_results]
comparison_df = pd.DataFrame([{
    "Model":    r["name"],
    "Accuracy": round(r["accuracy"], 4),
    "F1 Score": round(r["f1"], 4),
    "ROC-AUC":  round(r["roc_auc"], 4)
} for r in results]).sort_values("ROC-AUC", ascending=False)
print("\n", comparison_df.to_string(index=False))

# ─────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────
# ROC curves
plt.figure(figsize=(10, 7))
for r in results:
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    plt.plot(fpr, tpr,
             label=f"{r['name']} (AUC={r['roc_auc']:.4f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_PATH + "roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, r in enumerate(results):
    cm = confusion_matrix(y_test, r["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"])
    axes[i].set_title(r["name"])
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")
plt.tight_layout()
plt.savefig(PLOT_PATH + "confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# 8. SHAP
# ─────────────────────────────────────────────
X_test_sample = X_test[:500]

print("Computing SHAP for Logistic Regression...")
lr_explainer = shap.LinearExplainer(lr, X_train)
lr_shap = lr_explainer.shap_values(X_test_sample)
plt.figure()
shap.summary_plot(lr_shap, X_test_sample,
                  feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(PLOT_PATH + "shap_lr.png", dpi=150, bbox_inches="tight")
plt.close()

print("Computing SHAP for Random Forest...")
rf_explainer = shap.TreeExplainer(rf)
rf_shap = rf_explainer.shap_values(X_test_sample)
rf_shap_vals = rf_shap[1] if isinstance(rf_shap, list) else rf_shap
plt.figure()
shap.summary_plot(rf_shap_vals, X_test_sample,
                  feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(PLOT_PATH + "shap_rf.png", dpi=150, bbox_inches="tight")
plt.close()

print("Computing SHAP for XGBoost...")
xgb_explainer = shap.TreeExplainer(xgb)
xgb_shap = xgb_explainer.shap_values(X_test_sample)
plt.figure()
shap.summary_plot(xgb_shap, X_test_sample,
                  feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(PLOT_PATH + "shap_xgb.png", dpi=150, bbox_inches="tight")
plt.close()
print("SHAP plots saved")

# ─────────────────────────────────────────────
# 9. FIX XGBOOST BASE_SCORE AND SAVE
# ─────────────────────────────────────────────
# Fix base_score to avoid SHAP parsing errors in the app
booster = xgb.get_booster()
booster.save_model(SAVE_PATH + "xgb_temp.json")
with open(SAVE_PATH + "xgb_temp.json", "r") as f:
    model_json = json.load(f)
model_json["learner"]["learner_model_param"]["base_score"] = "0.5"
with open(SAVE_PATH + "xgb_fixed.json", "w") as f:
    json.dump(model_json, f)
os.remove(SAVE_PATH + "xgb_temp.json")
print("XGBoost base_score fixed")

# ─────────────────────────────────────────────
# 10. SAVE ALL
# ─────────────────────────────────────────────
with open(SAVE_PATH + "model_lr.pkl", "wb") as f:
    pickle.dump(lr, f)

with open(SAVE_PATH + "model_rf.pkl", "wb") as f:
    pickle.dump(rf, f)

with open(SAVE_PATH + "model_xgb.pkl", "wb") as f:
    pickle.dump(xgb, f)

with open(SAVE_PATH + "xgb_threshold.pkl", "wb") as f:
    pickle.dump(XGB_THRESHOLD, f)

comparison_df.to_csv(SAVE_PATH + "ml_comparison.csv", index=False)

print(f"\nAll models saved to {SAVE_PATH}")
for f in os.listdir(SAVE_PATH):
    print(f"  - {f}")