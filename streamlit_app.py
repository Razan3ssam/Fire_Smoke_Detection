import os
import glob
import time
import base64
import tempfile

import cv2
import streamlit as st

import config
from detect import Detector
from logger import log_event
from notifications import start_alarm_loop, stop_alarm_loop, play_alert, notify_owner

st.set_page_config(page_title="Fire & Smoke Detection", page_icon="🔥", layout="wide")

# ---------- Cached resources ----------
@st.cache_resource
def load_detector():
    return Detector()


@st.cache_data(show_spinner=False)
def _file_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _find_file(basename):
    """Finds a file like ground.jpg / ground.png / ground.webp in the project folder."""
    for ext in ("jpg", "jpeg", "png", "webp"):
        path = f"{basename}.{ext}"
        if os.path.exists(path):
            return path
    return None


detector = load_detector()

# ---------- Session state ----------
defaults = {
    "fire_count": 0,
    "smoke_count": 0,
    "alert_active": False,
    "alert_type": None,
    "last_confidence": 0.0,
    "cam_cap": None,
    "vid_cap": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------- Styling ----------
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
    .metric-card {
        background: rgba(0,0,0,0.55);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(8px);
        margin-bottom: 14px;
    }
    .metric-value { font-size: 1.8rem; font-weight: 800; color: #fff; }
    .metric-label { font-size: 0.8rem; color: #d8dee6; letter-spacing: 1px; }
    .stats-row { display: flex; gap: 14px; }
    .stats-row .metric-card { flex: 1; }
    .team-bar {
        display: flex; justify-content: center; gap: 28px; flex-wrap: wrap;
        padding: 18px; margin-top: 30px;
        background: rgba(0,0,0,0.45); border-radius: 16px;
    }
    .team-link {
        color: #fff; text-decoration: none; font-weight: 600; font-size: 0.95rem;
        display: flex; align-items: center; gap: 8px;
    }
    .team-link i { color: #0A66C2; font-size: 1.3rem; }
    .team-link:hover { color: #ffb84d; }
</style>
""", unsafe_allow_html=True)



st.markdown("""
<style>
[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"],
label, span, p, div, button {
    color:white !important;
}
div[role="radiogroup"] label{color:white !important;}
.stSlider label{color:white !important;}
.stCheckbox label{color:white !important;}
.stFileUploader label{color:white !important;}
</style>
""", unsafe_allow_html=True)

bg_placeholder = st.empty()


def render_background():
    """Ground image by default. Switches to the fire/smoke video while an alert is active."""
    video_path = None
    if st.session_state.alert_active:
        if st.session_state.alert_type == "fire" and os.path.exists(config.FIRE_VIDEO_PATH):
            video_path = config.FIRE_VIDEO_PATH
        elif st.session_state.alert_type == "smoke" and os.path.exists(config.SMOKE_VIDEO_PATH):
            video_path = config.SMOKE_VIDEO_PATH

    if video_path:
        video_b64 = _file_to_base64(video_path)
        bg_placeholder.markdown(f"""
        <style>.stApp {{ background: transparent; }}</style>
        <video autoplay muted loop id="bg-video"
            style="position:fixed; top:0; left:0; width:100%; height:100%;
                   object-fit:cover; z-index:-1; opacity:0.55;">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        """, unsafe_allow_html=True)
        return

    ground_path = _find_file(config.GROUND_IMAGE_BASENAME)
    if ground_path:
        img_b64 = _file_to_base64(ground_path)
        ext = ground_path.split(".")[-1]
        bg_placeholder.markdown(f"""
        <style>
        .stApp {{
            background: url("data:image/{ext};base64,{img_b64}") center center fixed;
            background-size: cover;
        }}
        </style>
        """, unsafe_allow_html=True)
    else:
        bg_placeholder.markdown("""
        <style>
        .stApp {
            background: linear-gradient(-45deg,#0b0f14,#10151c,#0b0f14,#131a22);
            background-size: 400% 400%;
            animation: gradientShift 6s ease infinite;
        }
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        </style>
        """, unsafe_allow_html=True)


render_background()

st.markdown(
    '<h1 style="color:#ffffff; font-weight:800; text-shadow: 0 2px 8px rgba(0,0,0,0.8);">'
    '🔥 Fire &amp; Smoke Detection</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#ffffff; text-shadow: 0 1px 6px rgba(0,0,0,0.8);">'
    'AI-Powered Real-Time Threat Detection</p>',
    unsafe_allow_html=True,
)

# ---------- Top bar: input source + confidence ----------
top_col1, top_col2 = st.columns([2, 1])
with top_col1:
    input_source = st.radio("Choose input source", ["Camera", "Upload Image", "Upload Video"],
                             horizontal=True, label_visibility="collapsed")
with top_col2:
    smoke_threshold = st.slider("Smoke sensitivity (lower = catches more, less certain)",
                                 0.1, 0.95, config.SMOKE_CONFIDENCE_THRESHOLD, 0.05)
    config.SMOKE_CONFIDENCE_THRESHOLD = smoke_threshold
    st.caption(f"🔥 Fire sensitivity is fixed low ({config.FIRE_CONFIDENCE_THRESHOLD*100:.0f}%) "
               f"so small flames still get caught even next to smoke.")

col_video, col_stats = st.columns([2, 1])

with col_video:
    frame_placeholder = st.empty()

with col_stats:
    status_placeholder = st.empty()
    dismiss_placeholder = st.empty()
    stats_placeholder = st.empty()


def render_status():
    risk_colors = {"LOW": "#34d399", "MEDIUM": "#ffb84d", "HIGH": "#ff4d4d"}
    if st.session_state.alert_active:
        label = st.session_state.alert_type.upper()
        status_placeholder.markdown(f"""
        <div class="metric-card" style="border-color:#ff4d4d;">
            <div class="metric-label">⚠ ALERT ACTIVE — EVENT: {label}</div>
            <div class="metric-value" style="color:#ff4d4d;">{label} DETECTED</div>
            <div class="metric-label">Confidence: {st.session_state.last_confidence*100:.1f}%</div>
            <div class="metric-label">Risk Level: {st.session_state.last_confidence*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
        if dismiss_placeholder.button("✅ I will take care of it", use_container_width=True):
            stop_alarm_loop()
            st.session_state.alert_active = False
            st.session_state.alert_type = None
            st.rerun()
    else:
        dismiss_placeholder.empty()
        status_placeholder.markdown("""
        <div class="metric-card">
            <div class="metric-label">STATUS</div>
            <div class="metric-value" style="color:#34d399;">No threat detected</div>
        </div>""", unsafe_allow_html=True)

    stats_placeholder.markdown(f"""
    <div class="stats-row">
        <div class="metric-card"><div class="metric-value">{st.session_state.fire_count}</div>
            <div class="metric-label">FIRE EVENTS</div></div>
        <div class="metric-card"><div class="metric-value">{st.session_state.smoke_count}</div>
            <div class="metric-label">SMOKE EVENTS</div></div>
    </div>""", unsafe_allow_html=True)


def handle_detections(detections):
    class_names_seen = set()
    for d in detections:
        log_event(d["class_name"], d["confidence"], d["risk"])
        class_names_seen.add(d["class_name"].lower())
        if d["class_name"].lower() == "fire":
            st.session_state.fire_count += 1
        elif d["class_name"].lower() == "smoke":
            st.session_state.smoke_count += 1

    if not st.session_state.alert_active and detections:
        top = max(detections, key=lambda d: d["confidence"])
        st.session_state.last_confidence = top["confidence"]

        if "fire" in class_names_seen:
            st.session_state.alert_active = True
            st.session_state.alert_type = "fire"
            start_alarm_loop()
            play_alert(f"Warning! Fire detected, {int(top['confidence']*100)} percent confidence.")
            if top["confidence"] >= 0.70:
                notify_owner(top["class_name"], top["confidence"], top["risk"])
        elif "smoke" in class_names_seen:
            st.session_state.alert_active = True
            st.session_state.alert_type = "smoke"
            play_alert(f"Warning! Smoke detected, {int(top['confidence']*100)} percent confidence.")
            if top["confidence"] >= 0.70:
                notify_owner(top["class_name"], top["confidence"], top["risk"])

        render_background()

    render_status()

render_status()

# ---------- Input handling ----------
if input_source == "Camera":
    run = st.checkbox("▶ Start Camera", key="cam_run")
    if run:
        if st.session_state.cam_cap is None:
            st.session_state.cam_cap = cv2.VideoCapture(config.CAMERA_INDEX)
        ret, frame = st.session_state.cam_cap.read()
        if ret:
            annotated_frame, detections, _ = detector.process_frame(frame)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            handle_detections(detections)
        time.sleep(0.05)
        st.rerun()
    else:
        if st.session_state.cam_cap is not None:
            st.session_state.cam_cap.release()
            st.session_state.cam_cap = None
        st.info("Tick **Start Camera** to begin monitoring.")

elif input_source == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded_image is not None:
        with open("temp_uploaded_image.jpg", "wb") as f:
            f.write(uploaded_image.read())
        frame = cv2.imread("temp_uploaded_image.jpg")
        annotated_frame, detections, _ = detector.process_frame(frame)
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        handle_detections(detections)

elif input_source == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    play = st.checkbox("▶ Play Video", key="vid_play")
    if uploaded_video is not None and play:
        if st.session_state.vid_cap is None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            st.session_state.vid_cap = cv2.VideoCapture(tfile.name)
        ret, frame = st.session_state.vid_cap.read()
        if ret:
            annotated_frame, detections, _ = detector.process_frame(frame)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            handle_detections(detections)
            time.sleep(0.03)
            st.rerun()
        else:
            st.success("Video finished.")
            st.session_state.vid_cap.release()
            st.session_state.vid_cap = None
    else:
        if st.session_state.vid_cap is not None:
            st.session_state.vid_cap.release()
            st.session_state.vid_cap = None

# ---------- Team / footer ----------
TEAM = [
    {"name": "Razan Essam", "url": "https://www.linkedin.com/in/razan-essam-1a5a0331a/"},
    {"name": "Fatma Mohamed", "url": "https://www.linkedin.com/in/fatma-mohamed-5025a5367/"},
    {"name": "Roaa Shoaib", "url": "https://www.linkedin.com/in/roaa-shoaib-501a4b36a/"},
]
links_html = "".join(
    f'<a class="team-link" href="{m["url"]}" target="_blank">'
    f'<i class="fa-brands fa-linkedin"></i> {m["name"]}</a>'
    for m in TEAM
)
st.markdown(f'<div class="team-bar">{links_html}</div>', unsafe_allow_html=True)