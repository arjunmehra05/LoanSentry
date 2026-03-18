"""
LoanSentry - Training Script 3: Neural Network
===============================================
Covers:
- Feedforward Neural Network with TensorFlow/Keras
- Training with early stopping and LR scheduling
- Threshold tuning
- SHAP explainability
- Saving model
"""

import os
import pickle
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, roc_auc_score,
                              roc_curve)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import callbacks, layers

warnings.filterwarnings("ignore")
print(f"TensorFlow version: {tf.__version__}")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LOAD_PATH = "output/"
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

print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

# ─────────────────────────────────────────────
# 2. BUILD MODEL
# ─────────────────────────────────────────────
def build_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_model(X_train.shape[1])
model.summary()

# ─────────────────────────────────────────────
# 3. CALLBACKS
# ─────────────────────────────────────────────
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss", patience=5,
    restore_best_weights=True, verbose=1)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5,
    patience=3, min_lr=1e-6, verbose=1)

model_checkpoint = callbacks.ModelCheckpoint(
    filepath=SAVE_PATH + "model_nn_best.keras",
    monitor="val_loss", save_best_only=True, verbose=1)

# ─────────────────────────────────────────────
# 4. CLASS WEIGHTS
# ─────────────────────────────────────────────
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

# ─────────────────────────────────────────────
# 5. TRAIN
# ─────────────────────────────────────────────
print("Training Neural Network...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=1024,
    validation_split=0.15,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)
print("Training complete")

# ─────────────────────────────────────────────
# 6. TRAINING HISTORY PLOT
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history["loss"],     label="Train Loss", linewidth=2)
axes[0].plot(history.history["val_loss"], label="Val Loss",   linewidth=2)
axes[0].set_title("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["accuracy"],     label="Train Accuracy", linewidth=2)
axes[1].plot(history.history["val_accuracy"], label="Val Accuracy",   linewidth=2)
axes[1].set_title("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Neural Network Training History", fontsize=14)
plt.tight_layout()
plt.savefig(PLOT_PATH + "nn_training_history.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# 7. THRESHOLD TUNING
# ─────────────────────────────────────────────
y_prob_nn = model.predict(X_test).flatten()

thresholds = np.arange(0.1, 0.6, 0.05)
f1_scores  = [f1_score(y_test, (y_prob_nn >= t).astype(int))
              for t in thresholds]

NN_THRESHOLD = thresholds[np.argmax(f1_scores)]
print(f"Best threshold: {NN_THRESHOLD:.2f} | Best F1: {max(f1_scores):.4f}")

plt.figure(figsize=(10, 5))
plt.plot(thresholds, f1_scores, marker="o", linewidth=2, color="steelblue")
plt.axvline(x=NN_THRESHOLD, color="tomato", linestyle="--",
            label=f"Best = {NN_THRESHOLD:.2f}")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("F1 Score vs Threshold - Neural Network")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_PATH + "nn_threshold.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# 8. EVALUATE
# ─────────────────────────────────────────────
y_pred_nn = (y_prob_nn >= NN_THRESHOLD).astype(int)
acc = accuracy_score(y_test, y_pred_nn)
f1  = f1_score(y_test, y_pred_nn)
roc = roc_auc_score(y_test, y_prob_nn)

print(f"\n{'='*40}")
print(f"Neural Network (threshold={NN_THRESHOLD:.2f})")
print(f"{'='*40}")
print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc:.4f}")
print(classification_report(y_test, y_pred_nn))

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob_nn)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, color="steelblue",
         label=f"Neural Network (AUC={roc:.4f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Neural Network")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_PATH + "nn_roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_nn)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"])
plt.title(f"Confusion Matrix - Neural Network (t={NN_THRESHOLD:.2f})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(PLOT_PATH + "nn_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# 9. SHAP
# ─────────────────────────────────────────────
print("Computing SHAP values for Neural Network...")
X_background   = X_train[:200]
X_test_sample  = X_test[:300]

nn_explainer   = shap.DeepExplainer(model, X_background)
nn_shap_values = nn_explainer.shap_values(X_test_sample)

nn_shap_vals = nn_shap_values[0] if isinstance(nn_shap_values, list) \
               else nn_shap_values
if nn_shap_vals.ndim == 3:
    nn_shap_vals = nn_shap_vals[:, :, 0]

plt.figure()
shap.summary_plot(nn_shap_vals, X_test_sample,
                  feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(PLOT_PATH + "shap_nn.png", dpi=150, bbox_inches="tight")
plt.close()
print("SHAP plot saved")

# ─────────────────────────────────────────────
# 10. SAVE
# ─────────────────────────────────────────────
model.save(SAVE_PATH + "model_nn.keras")

with open(SAVE_PATH + "nn_threshold.pkl", "wb") as f:
    pickle.dump(float(NN_THRESHOLD), f)

with open(SAVE_PATH + "nn_shap_values.pkl", "wb") as f:
    pickle.dump(nn_shap_vals, f)

print(f"\nAll files saved to {SAVE_PATH}")
for f in os.listdir(SAVE_PATH):
    print(f"  - {f}")