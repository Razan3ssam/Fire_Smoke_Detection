import cv2
import winsound
import time
import pyttsx3
import threading
from ultralytics import YOLO


# Load YOLO model
model = YOLO("best.pt")


# Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)


# Function for alarm + voice
def play_alert(speech):

    # Alarm sound
    winsound.PlaySound(
        "twisted-colossus-fire-alarm-414915.wav",
        winsound.SND_FILENAME | winsound.SND_ASYNC
    )

    # Voice warning
    engine.say(speech)
    engine.runAndWait()



# Open Camera
cap = cv2.VideoCapture(0)


# Settings
CONFIDENCE_THRESHOLD = 0.65
last_alert_time = 0
ALERT_COOLDOWN = 5   # seconds


print("The camera turned on... press 'r' to close it")


while True:

    ret, frame = cap.read()

    if not ret:
        print("There's no picture from the camera")
        break


    results = model(frame, verbose=False)


    alert_triggered = False
    highest_risk = "LOW"



    for result in results:

        boxes = result.boxes


        for box in boxes:

            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]


            if confidence >= CONFIDENCE_THRESHOLD:


                x1, y1, x2, y2 = map(int, box.xyxy[0])


                # Risk Level
                if confidence >= 0.90:

                    risk = "HIGH"
                    color = (0, 0, 255)


                elif confidence >= 0.80:

                    risk = "MEDIUM"
                    color = (0, 165, 255)


                else:

                    risk = "LOW"
                    color = (0, 255, 255)



                # Save highest risk
                if risk == "HIGH":

                    highest_risk = "HIGH"


                elif risk == "MEDIUM" and highest_risk != "HIGH":

                    highest_risk = "MEDIUM"



                # Bounding Box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )


                # Label
                label = f"{class_name} {confidence*100:.1f}% | Risk: {risk}"


                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )


                alert_triggered = True




    # Alert Section
    if alert_triggered:


        if highest_risk == "HIGH":

            warning_color = (0, 0, 255)
            message = "HIGH RISK FIRE!"
            speech = "Warning! High risk fire detected. Please evacuate immediately."



        elif highest_risk == "MEDIUM":

            warning_color = (0, 165, 255)
            message = "MEDIUM RISK!"
            speech = "Warning! Medium risk fire detected. Please check the area."



        else:

            warning_color = (0, 255, 255)
            message = "LOW RISK!"
            speech = "Warning! Low risk fire or smoke detected."



        cv2.putText(
            frame,
            message,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            warning_color,
            3
        )



        current_time = time.time()



        if current_time - last_alert_time >= ALERT_COOLDOWN:


            print(speech)


            threading.Thread(
                target=play_alert,
                args=(speech,),
                daemon=True
            ).start()


            last_alert_time = current_time





    cv2.imshow(
        "Fire & Smoke Detection",
        frame
    )



    if cv2.waitKey(1) & 0xFF == ord('r'):

        break




cap.release()
cv2.destroyAllWindows()