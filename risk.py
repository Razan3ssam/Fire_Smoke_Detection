from typing import List, Dict


def calculate_risk(
    has_fire: bool,
    has_smoke: bool,
    detections: List[Dict],
    image_shape: tuple,
) -> dict:
    """
    Calculate risk percentage based on detections.

    Logic:
    - Fire has higher weight than smoke
    - Larger area of detection → higher risk
    - Higher confidence → higher risk
    - Multiple detections increase risk
    """
    if not detections:
        return {
            "risk_percent": 0.0,
            "level": "SAFE",
            "color": "#22c55e",  # green
            "message": "No fire or smoke detected. Area is safe.",
        }

    h, w = image_shape[:2]
    total_area = max(h * w, 1)

    fire_score = 0.0
    smoke_score = 0.0

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        box_area = max((x2 - x1) * (y2 - y1), 1)
        area_ratio = box_area / total_area
        conf = det["confidence"]

        # Weighted contribution
        contribution = (area_ratio * 0.6 + conf * 0.4) * 100

        if det["label"].lower() == "fire":
            fire_score += contribution * 1.4  # fire is more dangerous
        else:
            smoke_score += contribution * 0.9

    # Cap individual scores
    fire_score = min(fire_score, 85)
    smoke_score = min(smoke_score, 60)

    # Combined risk
    risk = fire_score + smoke_score * 0.7
    risk = min(risk, 100.0)

    # Boost if both fire and smoke exist
    if has_fire and has_smoke:
        risk = min(risk * 1.15, 100.0)

    # Final classification
    if risk >= 75:
        level = "CRITICAL"
        color = "#dc2626"  # red
        message = "CRITICAL DANGER! Immediate evacuation recommended."
    elif risk >= 50:
        level = "HIGH"
        color = "#ea580c"  # orange
        message = "High risk detected. Investigate immediately."
    elif risk >= 25:
        level = "MEDIUM"
        color = "#eab308"  # yellow
        message = "Medium risk. Monitor the area closely."
    elif risk > 0:
        level = "LOW"
        color = "#84cc16"  # light green
        message = "Low risk detected. Stay alert."
    else:
        level = "SAFE"
        color = "#22c55e"
        message = "No fire or smoke detected. Area is safe."

    return {
        "risk_percent": round(risk, 1),
        "level": level,
        "color": color,
        "message": message,
        "fire_score": round(fire_score, 1),
        "smoke_score": round(smoke_score, 1),
    }