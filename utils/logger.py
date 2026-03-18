import csv
import os
from datetime import datetime

LOG_PATH = "logs/predictions_log.csv"

def log_prediction(input_dict, prediction_results):
    file_exists = os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "loan_amnt", "annual_inc", "dti",
                "fico_avg", "emp_length", "int_rate", "grade",
                "risk_score", "risk_category", "confidence"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            input_dict.get("loan_amnt", ""),
            input_dict.get("annual_inc", ""),
            input_dict.get("dti", ""),
            input_dict.get("FICO_AVG", ""),
            input_dict.get("emp_length", ""),
            input_dict.get("int_rate", ""),
            input_dict.get("grade", ""),
            prediction_results.get("prob_ensemble", ""),
            prediction_results.get("risk_category", ""),
            prediction_results.get("confidence", "")
        ])