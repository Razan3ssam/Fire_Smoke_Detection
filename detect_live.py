import cv2
import winsound  
import time
from ultralytics import YOLO

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

CONFIDENCE_THRESHOLD = 0.65
last_alert_time = 0
ALERT_COOLDOWN = 2  

print("The camera turned on... press 'r' to close it")

while True:
    ret, frame = cap.read()
    if not ret:
        print("There's no picture from the camera")
        break

    results = model(frame, verbose=False)

    alert_triggered = False

    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if confidence >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                label = f"{class_name} {confidence*100:.1f}%"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                alert_triggered = True

    if alert_triggered:
        cv2.putText(frame, "!! WARNING !!", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        current_time = time.time()
        if current_time - last_alert_time >= ALERT_COOLDOWN:
            print("Warning: Fire or smoke detected!")
            winsound.Beep(1000, 500)  
            last_alert_time = current_time

    cv2.imshow("Fire & Smoke Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('r'):
        break

cap.release()
cv2.destroyAllWindows()