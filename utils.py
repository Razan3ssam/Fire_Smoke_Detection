import cv2
import numpy as np
import base64
from PIL import Image
import io


def image_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) to RGB for Streamlit display"""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR"""
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(image_bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR to PIL Image"""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def resize_keep_aspect(image: np.ndarray, max_side: int = 1280) -> np.ndarray:
    """Resize image keeping aspect ratio"""
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_risk_banner(image: np.ndarray, risk_info: dict) -> np.ndarray:
    """Draw a risk banner on top of the image"""
    img = image.copy()
    h, w = img.shape[:2]

    level = risk_info["level"]
    percent = risk_info["risk_percent"]
    color_hex = risk_info["color"]

    # Convert hex to BGR
    color_hex = color_hex.lstrip("#")
    r, g, b = int(color_hex[0:2], 16), int(color_hex[2:4], 16), int(color_hex[4:6], 16)
    color = (b, g, r)

    banner_h = 50
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    text = f"RISK: {percent}%  |  {level}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (w - tw) // 2
    y = (banner_h + th) // 2
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness)

    return img


def generate_alarm_audio_base64(duration: float = 1.5, frequency: int = 880) -> str:
    """
    Generate a simple alarm beep as base64 WAV (no external files needed).
    Returns data URI ready for st.audio or HTML audio tag.
    """
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Two-tone alarm
    wave1 = 0.4 * np.sin(2 * np.pi * frequency * t)
    wave2 = 0.3 * np.sin(2 * np.pi * (frequency * 1.5) * t)
    # Amplitude modulation for beeping effect
    mod = 0.5 * (1 + np.sign(np.sin(2 * np.pi * 4 * t)))  # 4 Hz on/off
    audio = (wave1 + wave2) * mod
    audio = np.clip(audio, -1, 1)

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)

    # Create WAV in memory
    buffer = io.BytesIO()
    import wave
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"
