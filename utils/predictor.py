import pickle
import numpy as np
import tensorflow as tf
import shap

MODEL_PATH = "models/"

def load_models():
    with open(MODEL_PATH + "model_lr.pkl", "rb") as f:
        lr = pickle.load(f)
    with open(MODEL_PATH + "model_rf.pkl", "rb") as f:
        rf = pickle.load(f)
    with open(MODEL_PATH + "model_xgb.pkl", "rb") as f:
        xgb = pickle.load(f)
    nn = tf.keras.models.load_model(MODEL_PATH + "model_nn.keras")

    with open(MODEL_PATH + "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_PATH + "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    with open(MODEL_PATH + "xgb_threshold.pkl", "rb") as f:
        xgb_threshold = pickle.load(f)
    with open(MODEL_PATH + "nn_threshold.pkl", "rb") as f:
        nn_threshold = pickle.load(f)
    with open(MODEL_PATH + "ensemble_weights.pkl", "rb") as f:
        ensemble_weights = pickle.load(f)
    with open(MODEL_PATH + "ensemble_threshold.pkl", "rb") as f:
        ensemble_threshold = pickle.load(f)

    return {
        "lr":               lr,
        "rf":               rf,
        "xgb":              xgb,
        "xgb_booster":      xgb.get_booster(),
        "nn":               nn,
        "scaler":           scaler,
        "feature_names":    feature_names,
        "xgb_threshold":    xgb_threshold,
        "nn_threshold":     nn_threshold,
        "ensemble_weights": ensemble_weights,
        "ensemble_threshold": ensemble_threshold
    }


def predict(input_dict, models):
    feature_names = models["feature_names"]
    scaler        = models["scaler"]

    input_array  = np.array([[input_dict.get(f, 0) for f in feature_names]])
    input_scaled = scaler.transform(input_array)

    prob_lr  = models["lr"].predict_proba(input_scaled)[0][1]
    prob_rf  = models["rf"].predict_proba(input_scaled)[0][1]
    prob_xgb = models["xgb"].predict_proba(input_scaled)[0][1]
    prob_nn  = models["nn"].predict(input_scaled, verbose=0)[0][0]

    w             = models["ensemble_weights"]
    prob_ensemble = (
        w["lr"]  * prob_lr  +
        w["rf"]  * prob_rf  +
        w["xgb"] * prob_xgb +
        w["nn"]  * prob_nn
    )

    threshold  = models["ensemble_threshold"]
    prediction = int(prob_ensemble >= threshold)

    if prob_ensemble < 0.3:
        risk_category = "Low"
    elif prob_ensemble < 0.6:
        risk_category = "Medium"
    else:
        risk_category = "High"

    confidence = abs(prob_ensemble - threshold) / (1 - threshold) * 100
    confidence = min(round(confidence, 1), 100.0)

    return {
        "prob_ensemble": round(float(prob_ensemble), 4),
        "prob_lr":       round(float(prob_lr), 4),
        "prob_rf":       round(float(prob_rf), 4),
        "prob_xgb":      round(float(prob_xgb), 4),
        "prob_nn":       round(float(prob_nn), 4),
        "prediction":    prediction,
        "risk_category": risk_category,
        "confidence":    confidence,
        "input_scaled":  input_scaled
    }


def get_shap_values(input_scaled, models):
    explainer   = shap.TreeExplainer(models["xgb_booster"])
    shap_values = explainer.shap_values(input_scaled)
    return shap_values, models["feature_names"]