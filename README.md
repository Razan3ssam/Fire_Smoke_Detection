# Fire & Smoke Detection System

## Overview

Fire & Smoke Detection System is an AI-powered computer vision application developed for real-time fire and smoke detection. The system uses a custom-trained YOLO model to analyze live camera streams, uploaded images, and uploaded videos. It identifies fire and smoke, draws bounding boxes around detected objects, and displays both the model confidence score and the estimated risk level for each detection.

The project provides an interactive Streamlit interface for fast and user-friendly monitoring, making it suitable for educational, research, and intelligent surveillance applications.

---

## Features

- Real-time fire and smoke detection using a live camera.
- Upload and analyze images.
- Upload and analyze videos.
- Real-time streaming detection.
- Bounding box visualization.
- Confidence score for each detected object.
- Estimated risk level for each detection.
- Detection event logging.
- Interactive Streamlit web interface.
- Fast inference using a custom-trained YOLO model.

---

## Demo

### Fire and somke detection vedio

Watch the project demonstration video:

**[Fire and somke detection vedio](fire and somke detection vedio.mp4)**

The demo showcases:

- Live camera fire and smoke detection.
- Image upload and analysis.
- Video upload and frame-by-frame detection.
- Real-time streaming detection.
- Bounding boxes around detected fire and smoke.
- Confidence score for each detection.
- Estimated risk level displayed in real time.

---

## Dataset

The model was trained on a custom Fire & Smoke dataset:

https://universe.roboflow.com/nooda-alnujaifi/fire-and-smoke-bkm84/browse?queryText=&pageSize=50&startingIndex=50&browseQuery=true

---

## Training Notebook

Google Colab Notebook:

https://colab.research.google.com/drive/1lJ5k_iz_CUMmJa52AjmQ4Ii9OuPfoSDl?usp=sharing

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Razan3ssam/Fire_Smoke_Detection.git
```

Move to the project directory:

```bash
cd Fire_Smoke_Detection
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run streamlit_app.py
```

---

## How to Use

### Live Camera

Select **Camera** to perform real-time fire and smoke detection using your webcam.

### Image Detection

Select **Upload Image**, choose an image, and the system will detect fire or smoke, display bounding boxes, confidence scores, and the estimated risk level.

### Video Detection

Select **Upload Video** to analyze a video frame by frame and visualize the detection results throughout the video.

---

## Technologies

- Python
- YOLO
- OpenCV
- Streamlit
- Ultralytics
- NumPy
- Pandas

---

## Team

### Razan Essam

LinkedIn:
https://www.linkedin.com/in/razan-essam-1a5a0331a

### Roaa Shoaib

LinkedIn:
https://www.linkedin.com/in/roaa-shoaib-501a4b36a

### Fatma Mohamed

LinkedIn:
https://www.linkedin.com/in/fatma-mohamed-5025a5367

---

## License

This project is intended for educational and research purposes.
