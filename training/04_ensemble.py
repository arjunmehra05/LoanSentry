"""
LoanSentry - Training Script 4: Ensemble Model
===============================================
Covers:
- Soft voting, weighted voting, and stacking ensembles
- Full model comparison
- Saving ensemble weights and threshold
"""

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as MetaLearner
from sklearn.metrics import (accuracy_score, classification_report,
                              f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_predict
from tensorflow import keras

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LOAD_PATH = "output/"
SAVE_PATH = "output/"
PLOT_PATH = "plots/"

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(PLOT_PATH, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA AND MODELS
# ─────────────────────────────────────────────
X_train = np.load(LOAD_PATH + "X_train.npy")
X_test  = np.load(LOAD_PATH + "X_test.npy")
y_train = np.load(LOAD_PATH + "y_train.npy")
y_test  = np.load(LOAD_PATH + "y_test.npy")

with open(LOAD_PATH + "feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

with open(LOAD_PATH + "model_lr.pkl",  "rb") as f: lr  = pickle.load(f)
with open(LOAD_PATH + "model_rf.pkl",  "rb") as f: rf  = pickle.load(f)
with open(LOAD_PATH + "model_xgb.pkl", "rb") as f: xgb = pickle.load(f)

nn = keras.models.load_model(LOAD_PATH + "model_nn.keras")

with open(LOAD_PATH + "xgb_threshold.pkl", "rb") as f:
    xgb_threshold = pickle.load(f)
with open(LOAD_PATH + "nn_threshold.pkl", "rb") as f:
    nn_threshold = pickle.load(f)

print(f"XGBoost threshold: {xgb_threshold}")
print(f"Neural Network threshold: {nn_threshold}")
print("All models loaded")

# ─────────────────────────────────────────────
# 2. GET PROBABILITIES
# ─────────────────────────────────────────────
prob_lr  = lr.predict_proba(X_test)[:, 1]
prob_rf  = rf.predict_proba(X_test)[:, 1]
prob_xgb = xgb.predict_proba(X_test)[:, 1]
prob_nn  = nn.predict(X_test).flatten()

thresholds = np.arange(0.1, 0.6, 0.05)

# ─────────────────────────────────────────────
# 3. SOFT VOTING ENSEMBLE
# ─────────────────────────────────────────────
prob_voting   = (prob_lr + prob_rf + prob_xgb + prob_nn) / 4
f1_scores     = [f1_score(y_test, (prob_voting >= t).astype(int))
                 for t in thresholds]
t_voting      = thresholds[np.argmax(f1_scores)]
y_pred_voting = (prob_voting >= t_voting).astype(int)

voting_results = {
    "name":      "Soft Voting Ensemble",
    "y_pred":    y_pred_voting,
    "y_prob":    prob_voting,
    "accuracy":  accuracy_score(y_test, y_pred_voting),
    "f1":        f1_score(y_test, y_pred_voting),
    "roc_auc":   roc_auc_score(y_test, prob_voting),
    "threshold": t_voting
}
print(f"\nSoft Voting - F1: {voting_results['f1']:.4f} | ROC-AUC: {voting_results['roc_auc']:.4f}")

# ─────────────────────────────────────────────
# 4. WEIGHTED VOTING ENSEMBLE
# ─────────────────────────────────────────────
prob_weighted   = (0.15 * prob_lr + 0.20 * prob_rf +
                   0.35 * prob_xgb + 0.30 * prob_nn)
f1_scores_w     = [f1_score(y_test, (prob_weighted >= t).astype(int))
                   for t in thresholds]
t_weighted      = thresholds[np.argmax(f1_scores_w)]
y_pred_weighted = (prob_weighted >= t_weighted).astype(int)

weighted_results = {
    "name":      "Weighted Voting Ensemble",
    "y_pred":    y_pred_weighted,
    "y_prob":    prob_weighted,
    "accuracy":  accuracy_score(y_test, y_pred_weighted),
    "f1":        f1_score(y_test, y_pred_weighted),
    "roc_auc":   roc_auc_score(y_test, prob_weighted),
    "threshold": t_weighted
}
print(f"Weighted Voting - F1: {weighted_results['f1']:.4f} | ROC-AUC: {weighted_results['roc_auc']:.4f}")

# ─────────────────────────────────────────────
# 5. STACKING ENSEMBLE
# ─────────────────────────────────────────────
print("\nGenerating stacking meta-features (this takes a few minutes)...")

meta_lr  = cross_val_predict(lr,  X_train, y_train, cv=3, method="predict_proba")[:, 1]
meta_rf  = cross_val_predict(rf,  X_train, y_train, cv=3, method="predict_proba")[:, 1]
meta_xgb = cross_val_predict(xgb, X_train, y_train, cv=3, method="predict_proba")[:, 1]
meta_nn  = nn.predict(X_train).flatten()

X_meta_train = np.column_stack([meta_lr, meta_rf, meta_xgb, meta_nn])
X_meta_test  = np.column_stack([prob_lr, prob_rf, prob_xgb, prob_nn])

meta_learner = MetaLearner(max_iter=1000, random_state=42)
meta_learner.fit(X_meta_train, y_train)

prob_stacking   = meta_learner.predict_proba(X_meta_test)[:, 1]
f1_scores_s     = [f1_score(y_test, (prob_stacking >= t).astype(int))
                   for t in thresholds]
t_stacking      = thresholds[np.argmax(f1_scores_s)]
y_pred_stacking = (prob_stacking >= t_stacking).astype(int)

stacking_results = {
    "name":      "Stacking Ensemble",
    "y_pred":    y_pred_stacking,
    "y_prob":    prob_stacking,
    "accuracy":  accuracy_score(y_test, y_pred_stacking),
    "f1":        f1_score(y_test, y_pred_stacking),
    "roc_auc":   roc_auc_score(y_test, prob_stacking),
    "threshold": t_stacking
}
print(f"Stacking - F1: {stacking_results['f1']:.4f} | ROC-AUC: {stacking_results['roc_auc']:.4f}")

# ─────────────────────────────────────────────
# 6. FULL COMPARISON
# ─────────────────────────────────────────────
all_results = [
    {"name": "Logistic Regression",
     "y_prob": prob_lr, "y_pred": (prob_lr >= 0.5).astype(int),
     "accuracy": accuracy_score(y_test, (prob_lr >= 0.5).astype(int)),
     "f1": f1_score(y_test, (prob_lr >= 0.5).astype(int)),
     "roc_auc": roc_auc_score(y_test, prob_lr)},

    {"name": "Random Forest",
     "y_prob": prob_rf, "y_pred": (prob_rf >= 0.5).astype(int),
     "accuracy": accuracy_score(y_test, (prob_rf >= 0.5).astype(int)),
     "f1": f1_score(y_test, (prob_rf >= 0.5).astype(int)),
     "roc_auc": roc_auc_score(y_test, prob_rf)},

    {"name": f"XGBoost (t={xgb_threshold})",
     "y_prob": prob_xgb, "y_pred": (prob_xgb >= xgb_threshold).astype(int),
     "accuracy": accuracy_score(y_test, (prob_xgb >= xgb_threshold).astype(int)),
     "f1": f1_score(y_test, (prob_xgb >= xgb_threshold).astype(int)),
     "roc_auc": roc_auc_score(y_test, prob_xgb)},

    {"name": f"Neural Network (t={nn_threshold:.2f})",
     "y_prob": prob_nn, "y_pred": (prob_nn >= nn_threshold).astype(int),
     "accuracy": accuracy_score(y_test, (prob_nn >= nn_threshold).astype(int)),
     "f1": f1_score(y_test, (prob_nn >= nn_threshold).astype(int)),
     "roc_auc": roc_auc_score(y_test, prob_nn)},

    voting_results,
    weighted_results,
    stacking_results,
]

comparison_df = pd.DataFrame([{
    "Model":    r["name"],
    "Accuracy": round(r["accuracy"], 4),
    "F1 Score": round(r["f1"], 4),
    "ROC-AUC":  round(r["roc_auc"], 4),
} for r in all_results]).sort_values("ROC-AUC", ascending=False)

print("\n", comparison_df.to_string(index=False))

# ─────────────────────────────────────────────
# 7. ROC CURVES PLOT
# ─────────────────────────────────────────────
colors = ["steelblue", "seagreen", "tomato", "darkorange",
          "purple", "brown", "deeppink"]
plt.figure(figsize=(12, 8))
for i, r in enumerate(all_results):
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    plt.plot(fpr, tpr, label=f"{r['name']} (AUC={r['roc_auc']:.4f})",
             linewidth=2, color=colors[i])
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - All Models")
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_PATH + "roc_all_models.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# 8. SAVE
# ─────────────────────────────────────────────
ensemble_weights = {"lr": 0.15, "rf": 0.20, "xgb": 0.35, "nn": 0.30}
with open(SAVE_PATH + "ensemble_weights.pkl", "wb") as f:
    pickle.dump(ensemble_weights, f)

# Pick best ensemble by F1
best_ensemble = max([voting_results, weighted_results, stacking_results],
                    key=lambda x: x["f1"])
print(f"\nBest ensemble: {best_ensemble['name']}")

with open(SAVE_PATH + "ensemble_threshold.pkl", "wb") as f:
    pickle.dump(best_ensemble["threshold"], f)

with open(SAVE_PATH + "meta_learner.pkl", "wb") as f:
    pickle.dump(meta_learner, f)

comparison_df.to_csv(SAVE_PATH + "full_comparison.csv", index=False)

print(f"\nAll files saved to {SAVE_PATH}")
for f in os.listdir(SAVE_PATH):
    print(f"  - {f}")