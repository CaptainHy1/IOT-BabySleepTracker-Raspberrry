#!/usr/bin/env python3
import os
import time
import json
from datetime import datetime, timezone, timedelta

from .sensors.dht22 import read_dht22
from .sensors.mlx90614 import MLX90614
from .firebase_client import push_realtime

try:
    from .sensors.camera_supine_prone import SupineProne
except Exception:
    SupineProne = None

try:
    from .sensors.cry_detector import CryDetector
except Exception:
    CryDetector = None


def iso_vn_now():
    tz_vn = timezone(timedelta(hours=7))
    return datetime.now(tz=tz_vn).isoformat(timespec="seconds")


def _make_cry_from_env():
    """Create CryDetector from environment variables (all optional)."""
    if CryDetector is None:
        return None

    device = os.environ.get("CRY_DEVICE") or None
    tflite = os.environ.get("CRY_TFLITE") or None
    classes = os.environ.get("CRY_CLASSES") or None
    sr = int(os.environ.get("CRY_SR", "16000"))
    segment_sec = float(os.environ.get("CRY_SEG", "2.0"))
    hop_ratio = float(os.environ.get("CRY_HOP", "0.5"))
    cry_thr = float(os.environ.get("CRY_THR", "0.5"))
    min_dur = float(os.environ.get("CRY_MIN_DUR", "1.0"))
    rms_gate = float(os.environ.get("CRY_RMS_GATE", "0.001"))
    smooth_n = int(os.environ.get("CRY_SMOOTH", "3"))
    # use band-pass unless CRY_NO_BANDPASS is set (any non-empty value)
    use_bandpass = os.environ.get("CRY_NO_BANDPASS", "") == ""

    try:
        return CryDetector(
            tflite_path=tflite,
            classes_path=classes,
            sr=sr,
            segment_sec=segment_sec,
            hop_ratio=hop_ratio,
            cry_threshold=cry_thr,
            min_duration=min_dur,
            rms_gate=rms_gate,
            smooth_n=smooth_n,
            use_bandpass=use_bandpass,
            print_probs=False,
            device=device,
        )
    except Exception as e:
        print(f"[WARN] Cry detector init failed: {e}")
        return None


def _make_camera():
    if SupineProne is None:
        return None
    try:
        return SupineProne()
    except Exception as e:
        print(f"[WARN] Camera init failed: {e}")
        return None


def main():
    # print/push interval (seconds)
    PRINT_INTERVAL = float(os.environ.get("PRINT_INTERVAL", "5.0"))
    # latch seconds to keep isCrying true after a detection
    CRY_HOLD_SEC = float(os.environ.get("CRY_HOLD_SEC", "3.0"))
    cry_latch_until = 0.0

    # IR temperature sensor
    try:
        mlx = MLX90614()
    except Exception as e:
        print(f"[WARN] MLX90614 init failed: {e}")
        mlx = None

    # camera and mic
    cam = _make_camera()
    cry = _make_cry_from_env()

    print("[INFO] Running. Press Ctrl+C to stop.")
    last_pose = None
    next_print = time.time()

    try:
        while True:
            now = time.time()

            # DHT22
            room_temp, room_hum = read_dht22()

            # MLX90614
            if mlx is not None:
                try:
                    baby_temp = round(mlx.read_object(), 2)
                    env_temp = round(mlx.read_ambient(), 2)
                except Exception:
                    baby_temp = None
                    env_temp = None
            else:
                baby_temp = None
                env_temp = None

            # camera pose
            if cam is not None:
                try:
                    pose = cam.read_position()
                    if pose:
                        last_pose = pose
                except Exception as e:
                    print(f"[WARN] Camera read failed: {e}")
                    try:
                        cam.close()
                    except Exception:
                        pass
                    cam = None

            # mic / cry-detector
            if cry is not None:
                try:
                    cry_state = cry.read_state()
                    if cry_state:
                        # simple debug line (human readable)
                        print({
                            "dbg": {
                                "pCry": cry_state.get("pCry"),
                                "smooth": cry_state.get("smooth"),
                                "rms": cry_state.get("rms"),
                                "event": cry_state.get("event"),
                            }
                        })
                        if cry_state.get("isCrying") or cry_state.get("event"):
                            cry_latch_until = max(cry_latch_until, now + CRY_HOLD_SEC)

                    is_crying = (now < cry_latch_until)
                except Exception as e:
                    print(f"[WARN] Cry read failed: {e}")
                    try:
                        cry.stop()
                    except Exception:
                        pass
                    cry = None
                    is_crying = False
            else:
                is_crying = False

            # print and push on interval
            if now >= next_print:
                status = "crying" if is_crying else "sleeping"
                payload = {
                    "babyTemperature": baby_temp,
                    "environmentHumidity": room_hum,
                    "environmentTemperature": env_temp,
                    "isCrying": is_crying,
                    "sleepPosition": last_pose or "unknown",
                    "status": status,
                    "temperature": room_temp,
                    "timestamp": iso_vn_now(),
                }

                print(json.dumps(payload, ensure_ascii=True))
                try:
                    push_realtime(payload)
                except Exception as e:
                    print(f"[WARN] Firebase push failed: {e}")

                next_print = now + PRINT_INTERVAL

            # allow audio stream to flow and keep CPU reasonable
            time.sleep(0.2)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            if cry is not None:
                cry.stop()
        except Exception:
            pass

        try:
            if cam is not None:
                cam.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
