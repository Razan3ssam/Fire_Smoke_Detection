"""
Event logger: appends every detection event to events.csv
"""

import csv
import os
from datetime import datetime
import config


def log_event(class_name, confidence, risk):
    file_exists = os.path.isfile(config.EVENTS_CSV_PATH)
    with open(config.EVENTS_CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Date", "Time", "Type", "Confidence", "Risk"])
        now = datetime.now()
        writer.writerow([
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            class_name,
            f"{confidence*100:.1f}%",
            risk,
        ])
