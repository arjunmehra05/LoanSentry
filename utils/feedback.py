import csv
import os
from datetime import datetime

FEEDBACK_PATH = "feedback_log.csv"

def log_feedback(risk_category, prob, feedback):
    file_exists = os.path.exists(FEEDBACK_PATH)

    with open(FEEDBACK_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "risk_category", 
                             "risk_score", "feedback"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            risk_category,
            prob,
            feedback
        ])