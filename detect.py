"""
Detection module: wraps the YOLO model, classifies risk level per detection,
and draws the bounding boxes / labels onto each frame.
"""

import cv2
from ultralytics import YOLO
import config

# The model's own names dict sometimes just contains numbers ("0", "1")
# instead of readable labels. Map class index -> display name here.
# IMPORTANT: check your dataset's data.yaml "names:" list to confirm the
# order, and swap these two if they come out reversed.
CLASS_ID_TO_NAME = {
    0: "Fire",
    1: "Smoke",
}


class Detector:
    def __init__(self):
        self.model = YOLO(config.MODEL_PATH)

    def _risk_level(self, confidence):
        if confidence >= config.RISK_HIGH:
            return "HIGH", (0, 0, 255)
        elif confidence >= config.RISK_MEDIUM:
            return "MEDIUM", (0, 165, 255)
        return "LOW", (0, 255, 255)

    def process_frame(self, frame):
        """
        Runs detection on a single frame.
        Returns: (annotated_frame, detections, highest_risk)
        detections is a list of dicts: {class_name, confidence, risk}
        """
        # Run inference at the lowest threshold we care about (fire's), then
        # filter each box against its own class-specific threshold below —
        # this way a faint/small fire isn't discarded before we even see it.
        min_conf = min(config.FIRE_CONFIDENCE_THRESHOLD, config.SMOKE_CONFIDENCE_THRESHOLD)
        results = self.model(frame, verbose=False, conf=min_conf)
        detections = []
        highest_risk = "LOW"
        risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = CLASS_ID_TO_NAME.get(class_id, self.model.names[class_id])

                class_threshold = (
                    config.FIRE_CONFIDENCE_THRESHOLD
                    if class_name.lower() == "fire"
                    else config.SMOKE_CONFIDENCE_THRESHOLD
                )
                if confidence < class_threshold:
                    continue

                risk, color = self._risk_level(confidence)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {confidence*100:.1f}% | {risk}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                detections.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "risk": risk,
                })

                if risk_order[risk] > risk_order[highest_risk]:
                    highest_risk = risk

        return frame, detections, highest_risk
