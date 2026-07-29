import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import base64
from pathlib import Path

from detector import FireSmokeDetector
from risk import calculate_risk
from utils import (
    image_to_rgb,
    pil_to_cv2,
    resize_keep_aspect,
    draw_risk_banner,
    generate_alarm_audio_base64,
)

# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🔥 FIRE & SMOKE RISK DETECTOR",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = Path(__file__).parent / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════
@st.cache_resource
def load_detector(model_path: str, conf: float):
    return FireSmokeDetector(model_path=model_path, conf_threshold=conf)


# ══════════════════════════════════════════════════════════
# ALARM
# ══════════════════════════════════════════════════════════
ALARM_FILE = Path(__file__).parent / "twisted-colossus-fire-alarm-414915.wav"


def play_alarm():
    if ALARM_FILE.exists():
        st.audio(str(ALARM_FILE), format="audio/wav")
        b64 = base64.b64encode(ALARM_FILE.read_bytes()).decode()
        st.markdown(
            f'<audio autoplay><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>',
            unsafe_allow_html=True,
        )
    else:
        uri = generate_alarm_audio_base64(duration=1.8, frequency=920)
        st.markdown(
            f'<audio autoplay><source src="{uri}" type="audio/wav"></audio>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔥 CONTROL PANEL")
    st.markdown("---")

    st.markdown("### ⚙️ Detection Settings")
    model_path = st.text_input("Model File", value="best (1).pt")
    conf_threshold = st.slider("Confidence", 0.15, 0.85, 0.35, 0.05)
    alarm_threshold = st.slider("Alarm Threshold %", 20, 90, 45, 5)

    st.markdown("---")
    st.markdown("### 📷 Input Source")
    source = st.radio(
        "Select Mode",
        ["🖼  Upload Image", "🎬  Upload Video", "📹  Live Webcam"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🔗 Team LinkedIn")
    li_name1 = st.text_input("Name 1", value="", placeholder="e.g. Ahmed Mohamed")
    li_url1  = st.text_input("LinkedIn URL 1", value="", placeholder="https://linkedin.com/in/...")
    li_name2 = st.text_input("Name 2", value="", placeholder="e.g. Sara Ali")
    li_url2  = st.text_input("LinkedIn URL 2", value="", placeholder="https://linkedin.com/in/...")
    li_name3 = st.text_input("Name 3", value="", placeholder="e.g. Mahmoud Hassan")
    li_url3  = st.text_input("LinkedIn URL 3", value="", placeholder="https://linkedin.com/in/...")

    st.markdown("---")
    st.markdown(
        """
        <div style="background:#1a0a00;border:1px solid #5c2a00;border-radius:10px;padding:12px;font-size:0.85rem;color:#e8c48a;">
        <b>Required files:</b><br>
        • best (1).pt<br>
        • twisted-colossus-fire-alarm-414915.wav
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="hero-banner">
        <h1>🔥 FIRE & SMOKE RISK DETECTOR</h1>
        <div class="subtitle">AI-Powered Threat Assessment System</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════
# LOAD DETECTOR
# ══════════════════════════════════════════════════════════
try:
    detector = load_detector(model_path, conf_threshold)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


# ══════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════
def process_frame(frame_bgr: np.ndarray):
    frame_bgr = resize_keep_aspect(frame_bgr, max_side=1280)
    result = detector.detect(frame_bgr)
    risk_info = calculate_risk(
        has_fire=result["has_fire"],
        has_smoke=result["has_smoke"],
        detections=result["detections"],
        image_shape=frame_bgr.shape,
    )
    annotated = draw_risk_banner(result["annotated_image"], risk_info)
    return result, risk_info, annotated


def show_risk_panel(risk_info: dict, result: dict):
    level = risk_info["level"]
    css_map = {
        "CRITICAL": "risk-critical",
        "HIGH": "risk-high",
        "MEDIUM": "risk-medium",
        "LOW": "risk-low",
        "SAFE": "risk-safe",
    }
    css_class = css_map.get(level, "risk-safe")

    st.markdown(
        f"""
        <div class="risk-card {css_class}">
            <h2>⚠ RISK LEVEL — {level}</h2>
            <h1>{risk_info['risk_percent']}%</h1>
            <p>{risk_info['message']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(min(risk_info["risk_percent"] / 100.0, 1.0))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("FIRE", "🔥 YES" if result["has_fire"] else "— NO")
    with c2:
        st.metric("SMOKE", "💨 YES" if result["has_smoke"] else "— NO")
    with c3:
        st.metric("CONFIDENCE", f"{result['max_conf']*100:.0f}%")
    with c4:
        st.metric("OBJECTS", len(result["detections"]))

    if result["detections"]:
        with st.expander("📋 Detection Details"):
            for i, det in enumerate(result["detections"], 1):
                st.markdown(
                    f"**{i}.** `{det['label']}` &nbsp;·&nbsp; "
                    f"Confidence **{det['confidence']:.2f}** &nbsp;·&nbsp; "
                    f"Box `{det['box']}`"
                )


# ══════════════════════════════════════════════════════════
# IMAGE MODE
# ══════════════════════════════════════════════════════════
if "Upload Image" in source:
    uploaded = st.file_uploader(
        "Drop an image here",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        pil_img = Image.open(uploaded)
        frame = pil_to_cv2(pil_img)

        with st.spinner("🔍 Scanning for threats..."):
            result, risk_info, annotated = process_frame(frame)

        col_left, col_right = st.columns([1.35, 1])

        with col_left:
            st.markdown("### Detection Result")
            st.image(image_to_rgb(annotated), use_container_width=True)

        with col_right:
            st.markdown("### Threat Assessment")
            show_risk_panel(risk_info, result)

            if risk_info["risk_percent"] >= alarm_threshold:
                st.markdown(
                    """
                    <div style="
                        background:linear-gradient(90deg,#7f0000,#cc0000);
                        color:white;text-align:center;padding:14px;
                        border-radius:10px;font-family:Orbitron,sans-serif;
                        font-weight:700;letter-spacing:2px;font-size:1.15rem;
                        box-shadow:0 0 25px rgba(255,0,0,0.5);
                        animation: criticalPulse 1s infinite;
                    ">
                        🚨 ALARM TRIGGERED — DANGER DETECTED
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                play_alarm()
            else:
                st.markdown(
                    """
                    <div style="
                        background:linear-gradient(90deg,#14532d,#16a34a);
                        color:white;text-align:center;padding:12px;
                        border-radius:10px;font-weight:700;letter-spacing:1px;
                    ">
                        ✅ ALL CLEAR — Risk below threshold
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════
# VIDEO MODE
# ══════════════════════════════════════════════════════════
elif "Upload Video" in source:
    uploaded_video = st.file_uploader(
        "Drop a video here",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )

    if uploaded_video is not None:
        tfile = Path("temp_video.mp4")
        tfile.write_bytes(uploaded_video.read())

        cap = cv2.VideoCapture(str(tfile))
        if not cap.isOpened():
            st.error("Could not open video file.")
            st.stop()

        stframe = st.empty()
        risk_box = st.empty()
        stop_btn = st.button("⏹ STOP VIDEO")

        frame_idx = 0
        process_every = 3

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % process_every != 0:
                continue

            result, risk_info, annotated = process_frame(frame)
            stframe.image(image_to_rgb(annotated), use_container_width=True)

            with risk_box.container():
                show_risk_panel(risk_info, result)
                if risk_info["risk_percent"] >= alarm_threshold:
                    play_alarm()

            time.sleep(0.03)

        cap.release()
        if tfile.exists():
            tfile.unlink()


# ══════════════════════════════════════════════════════════
# WEBCAM MODE
# ══════════════════════════════════════════════════════════
elif "Live Webcam" in source:
    st.markdown(
        """
        <div style="background:#1a0a00;border:1px solid #5c2a00;border-radius:10px;
                    padding:12px 16px;color:#e8c48a;margin-bottom:1rem;">
            ⚠ Live webcam works best when running <b>locally</b>.
            Click the checkbox below to start the camera feed.
        </div>
        """,
        unsafe_allow_html=True,
    )

    run = st.checkbox("▶ START LIVE DETECTION", value=False)

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot access webcam. Check if it's connected or used by another app.")
            st.stop()

        stframe = st.empty()
        risk_box = st.empty()
        stop = st.button("⏹ STOP CAMERA")

        while run and not stop:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture frame.")
                break

            result, risk_info, annotated = process_frame(frame)
            stframe.image(image_to_rgb(annotated), use_container_width=True)

            with risk_box.container():
                show_risk_panel(risk_info, result)
                if risk_info["risk_percent"] >= alarm_threshold:
                    play_alarm()

            time.sleep(0.04)

        cap.release()


# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
# Build LinkedIn section
linkedin_items = []
for name, url in [(li_name1, li_url1), (li_name2, li_url2), (li_name3, li_url3)]:
    if name and name.strip() and url and url.strip():
        safe_name = name.strip()
        safe_url = url.strip()
        linkedin_items.append(
            f'<a href="{safe_url}" target="_blank" '
            f'style="color:#0a66c2;text-decoration:none;font-weight:700;'
            f'margin:0 12px;font-size:0.95rem;">🔗 {safe_name}</a>'
        )

st.markdown("---")

if linkedin_items:
    links_html = " &nbsp;|&nbsp; ".join(linkedin_items)
    st.markdown(
        f'<div style="text-align:center;margin-bottom:10px;">'
        f'<div style="color:#c4a574;font-size:0.8rem;letter-spacing:2px;margin-bottom:6px;">TEAM</div>'
        f'{links_html}</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style="text-align:center;color:#6b4c2a;font-size:0.85rem;letter-spacing:1px;">
        FIRE & SMOKE RISK DETECTION SYSTEM &nbsp;·&nbsp; Powered by YOLOv8 + Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)