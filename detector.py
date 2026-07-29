import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


class FireSmokeDetector:
    """YOLOv8 based Fire & Smoke Detector"""

    def __init__(self, model_path: str = "best (1).pt", conf_threshold: float = 0.35):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. "
                "Please put 'best (1).pt' in the same folder as app.py"
            )
        self.model = YOLO(str(self.model_path))
        self.conf_threshold = conf_threshold

        # Common class names used in fire/smoke datasets
        self.class_map = {
            0: "Fire",
            1: "Smoke",
            "fire": "Fire",
            "smoke": "Smoke",
            "Fire": "Fire",
            "Smoke": "Smoke",
        }

    def detect(self, image: np.ndarray) -> dict:
        """
        Run detection on a BGR image.

        Returns
        -------
        dict with keys:
            - annotated_image : np.ndarray (BGR)
            - detections      : list of dicts
            - has_fire        : bool
            - has_smoke       : bool
            - max_conf        : float
            - fire_boxes      : list
            - smoke_boxes     : list
        """
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            verbose=False,
            imgsz=640,
        )

        annotated = image.copy()
        detections = []
        has_fire = False
        has_smoke = False
        max_conf = 0.0
        fire_boxes = []
        smoke_boxes = []

        if results and len(results) > 0:
            result = results[0]
            names = result.names  # class id -> name

            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)

                    raw_name = names.get(cls_id, str(cls_id))
                    label = self.class_map.get(raw_name, self.class_map.get(cls_id, raw_name))

                    x1, y1, x2, y2 = xyxy
                    color = (0, 0, 255) if label.lower() == "fire" else (0, 165, 255)  # red / orange

                    # Draw box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                    # Label background
                    text = f"{label} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(
                        annotated,
                        text,
                        (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    det = {
                        "label": label,
                        "confidence": conf,
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                    }
                    detections.append(det)
                    max_conf = max(max_conf, conf)

                    if label.lower() == "fire":
                        has_fire = True
                        fire_boxes.append(det)
                    elif label.lower() == "smoke":
                        has_smoke = True
                        smoke_boxes.append(det)

        return {
            "annotated_image": annotated,
            "detections": detections,
            "has_fire": has_fire,
            "has_smoke": has_smoke,
            "max_conf": max_conf,
            "fire_boxes": fire_boxes,
            "smoke_boxes": smoke_boxes,
        }