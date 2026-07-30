"""
Alarm sound + voice warning, plus a placeholder hook (notify_owner)
you can later connect to SMS / email / push notifications.
Emergency-service calling is intentionally NOT implemented here.
"""

import time
import threading
import winsound
import pyttsx3
import config

_engine = pyttsx3.init()
_engine.setProperty("rate", config.VOICE_RATE)


def _play(speech_text):
    try:
        winsound.PlaySound(config.ALARM_SOUND_PATH,
                            winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception as e:
        print(f"Could not play alarm sound: {e}")

    _engine.say(speech_text)
    _engine.runAndWait()


def play_alert(speech_text):
    """Plays the alarm sound + speaks the warning, in a background thread."""
    threading.Thread(target=_play, args=(speech_text,), daemon=True).start()


def notify_owner(class_name, confidence, risk):
    """
    Placeholder for your own notification integration
    (e.g. SMS via Twilio, email, push notification, etc.)
    Connect your own service here later. Not implemented by default.
    """
    print(f"[notify_owner] {class_name} detected - {confidence*100:.1f}% - risk {risk}")


_alarm_playing = False
_alarm_lock = threading.Lock()


def _run_alarm(duration):
    global _alarm_playing
    try:
        winsound.PlaySound(config.ALARM_SOUND_PATH,
                            winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
    except Exception as e:
        print(f"Could not play alarm sound: {e}")
    time.sleep(duration)
    winsound.PlaySound(None, winsound.SND_PURGE)
    _alarm_playing = False


def start_fire_siren(duration=None):
    """Starts the looping siren for `duration` seconds (default from config).
    Does nothing if the siren is already sounding."""
    global _alarm_playing
    duration = duration or config.ALARM_DURATION_SECONDS
    with _alarm_lock:
        if not _alarm_playing:
            _alarm_playing = True
            threading.Thread(target=_run_alarm, args=(duration,), daemon=True).start()


def start_siren_indefinite():
    """Starts the looping siren; keeps playing until stop_siren() is called."""
    global _alarm_playing
    with _alarm_lock:
        if not _alarm_playing:
            _alarm_playing = True
            try:
                winsound.PlaySound(config.ALARM_SOUND_PATH,
                                    winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
            except Exception as e:
                print(f"Could not play alarm sound: {e}")


def stop_siren():
    """Stops the siren immediately."""
    global _alarm_playing
    winsound.PlaySound(None, winsound.SND_PURGE)
    _alarm_playing = False


def stop_alarm_loop():
    """Immediately stops the looping siren (used by the dismiss button)."""
    global _alarm_playing
    try:
        winsound.PlaySound(None, winsound.SND_PURGE)
    except Exception as e:
        print(f"Could not stop alarm sound: {e}")
    _alarm_playing = False


def start_alarm_loop():
    """Starts a siren that loops forever until stop_alarm_loop() is called."""
    global _alarm_playing
    with _alarm_lock:
        if not _alarm_playing:
            _alarm_playing = True
            try:
                winsound.PlaySound(config.ALARM_SOUND_PATH,
                                    winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
            except Exception as e:
                print(f"Could not play alarm sound: {e}")
