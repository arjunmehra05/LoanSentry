import numpy as np
from sklearn.metrics import roc_curve, auc

def get_all_model_probs(input_scaled, models):
    prob_lr  = models["lr"].predict_proba(input_scaled)[0][1]
    prob_rf  = models["rf"].predict_proba(input_scaled)[0][1]
    prob_xgb = models["xgb"].predict_proba(input_scaled)[0][1]
    prob_nn  = models["nn"].predict(input_scaled, verbose=0)[0][0]

    w = models["ensemble_weights"]
    prob_ensemble = (
        w["lr"]  * prob_lr  +
        w["rf"]  * prob_rf  +
        w["xgb"] * prob_xgb +
        w["nn"]  * prob_nn
    )

    return {
        "Logistic Regression": round(float(prob_lr), 4),
        "Random Forest":       round(float(prob_rf), 4),
        "XGBoost":             round(float(prob_xgb), 4),
        "Neural Network":      round(float(prob_nn), 4),
        "Ensemble":            round(float(prob_ensemble), 4)
    }