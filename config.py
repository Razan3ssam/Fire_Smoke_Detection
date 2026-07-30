"""
Central configuration for the Fire & Smoke Monitoring System.
Change values here instead of digging through the rest of the code.
"""

MODEL_PATH = "best.pt"
CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 0.65

# Fire gets a lower bar than smoke, so even small/faint flames still get marked
FIRE_CONFIDENCE_THRESHOLD = 0.35
SMOKE_CONFIDENCE_THRESHOLD = 0.65

# Risk level thresholds (based on detection confidence)
RISK_HIGH = 0.90
RISK_MEDIUM = 0.80

ALARM_SOUND_PATH = "twisted-colossus-fire-alarm-414915.wav"
ALERT_COOLDOWN_SECONDS = 5
VOICE_RATE = 150

SCREENSHOT_DIR = "screenshots"
EVENTS_CSV_PATH = "events.csv"

# Model class name -> display name (edit if your model uses different labels)
CLASS_DISPLAY_NAMES = {
    "Fire": "Fire",
    "Smoke": "Smoke",
}

# Background videos shown once fire/smoke is detected (place these video files
# in the project folder with these exact names)
FIRE_VIDEO_PATH = "fire_vedio.mp4"
SMOKE_VIDEO_PATH = "smoke_vedio.mp4"

# How long the fire siren keeps sounding once triggered (seconds)
ALARM_DURATION_SECONDS = 30

# Default static background image shown when nothing is detected.
# Place a file named "ground" with one of these extensions in the project folder.
GROUND_IMAGE_CANDIDATES = ["ground.jpg", "ground.jpeg", "ground.png"]

# Static default background image (place a file named ground.jpg / ground.png
# in the project folder)
GROUND_IMAGE_BASENAME = "ground"

# Static default background image (place a file named "ground.jpg"/"ground.png"
# in the project folder)
GROUND_IMAGE_BASENAME = "ground"
