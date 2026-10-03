#!/usr/bin/env python3
"""
Test harness for CryDetector:
- Live from mic (default)
- Or from a WAV file (--wav)
Prints JSON-like lines with: isCrying, pCry, pNot, smooth, rms, event.
"""

import os, sys, time, json, argparse
from pathlib import Path

# --- import CryDetector from your codebase ---
HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]  # -> /baby-monitor-pi/baby-monitor-pi
sys.path.insert(0, str(ROOT))

from src.sensors import cry_detector as cd  # your provided file

def list_devices():
    import sounddevice as sd
    devs = sd.query_devices()
    print("\n=== Audio input devices ===")
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            print(f"[{i}] {d['name']}")
    print("===========================\n")

def run_from_mic(args):
    tfl = args.tflite or os.environ.get("CRY_TFLITE")
    cla = args.classes or os.environ.get("CRY_CLASSES")

    det = cd.CryDetector(
        tflite_path=tfl,
        classes_path=cla,
        sr=args.sr,
        segment_sec=args.segment_sec,
        hop_ratio=args.hop_ratio,
        cry_threshold=args.cry_thr,
        min_duration=args.min_dur,
        rms_gate=args.rms_gate,
        smooth_n=args.smooth_n,
        use_bandpass=not args.no_bandpass,
        print_probs=args.print_probs,
        device=args.device
    )
    print("[INFO] Listening from mic. Press Ctrl+C to stop.\n")

    t0 = time.time()
    try:
        while True:
            st = det.read_state()
            if st is None:
                time.sleep(0.01)
                continue
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            st["t"] = stamp
            print(json.dumps(st))
            if args.limit_sec and (time.time() - t0) >= args.limit_sec:
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        det.stop()
        print("\n[INFO] Stopped.")

def segmenter(y, seg_len, hop_len):
    N = len(y); i = 0
    while i + seg_len <= N:
        yield y[i:i+seg_len]
        i += hop_len

def run_from_wav(args):
    import numpy as np
    import librosa

    tfl = args.tflite or os.environ.get("CRY_TFLITE")
    cla = args.classes or os.environ.get("CRY_CLASSES")

    y, sr = librosa.load(args.wav, sr=args.sr, mono=True)
    print(f"[INFO] Loaded WAV: {args.wav} | samples={len(y)} | sr={sr}")


    det = cd.CryDetector(
        tflite_path=tfl,
        classes_path=cla,
        sr=args.sr,
        segment_sec=args.segment_sec,
        hop_ratio=args.hop_ratio,
        cry_threshold=args.cry_thr,
        min_duration=args.min_dur,
        rms_gate=args.rms_gate,
        smooth_n=args.smooth_n,
        use_bandpass=not args.no_bandpass,
        print_probs=args.print_probs,
        device=None
    )
    det.stop()  # stop mic, we feed from file

    seg_len = int(args.segment_sec * args.sr)
    hop_len = int(args.hop_ratio * seg_len)
    n_events = 0
    for seg in segmenter(y, seg_len, hop_len):
        import numpy as np
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms < det.rms_gate:
            pcry, pnot = 0.0, 1.0
            det.hist.append(pcry)
            ps = float(np.mean(det.hist))
            isCrying = ps >= det.cry_threshold
            above = False
        else:
            img = cd._mel_image(seg, sr=det.sr, target_hw=(128, 128),
                                n_fft=1024, hop_length=256, n_mels=128,
                                use_bandpass=det.use_bandpass)
            probs = det._infer_probs(img)
            if det.cry_idx is not None and det.not_idx is not None:
                pcry = float(probs[det.cry_idx]); pnot = float(probs[det.not_idx])
            elif probs.size == 2:
                if det.cry_idx is not None:
                    pcry = float(probs[det.cry_idx]); pnot = 1.0 - pcry
                else:
                    pcry = float(probs[1]); pnot = float(probs[0])
            elif det.cry_idx is not None:
                pcry = float(probs[det.cry_idx]); pnot = 1.0 - pcry
            else:
                m = int(np.argmax(probs)); pcry = float(probs[m]); pnot = 1.0 - pcry

            det.hist.append(pcry)
            ps = float(np.mean(det.hist))
            isCrying = ps >= det.cry_threshold
            above = isCrying

        if above:
            det.above_streak_s += det.frame_hop_s
        else:
            det.above_streak_s = max(0.0, det.above_streak_s - det.frame_hop_s)

        event = (det.above_streak_s >= det.min_duration)
        if event:
            det.above_streak_s = 0.0
            n_events += 1

        out = {"isCrying": isCrying, "pCry": round(pcry, 4), "pNot": round(pnot, 4),
               "smooth": round(ps, 4), "rms": round(rms, 6), "event": event}
        print(json.dumps(out))

    print(f"[INFO] Done. Total events: {n_events}")


def main():
    ap = argparse.ArgumentParser(description="Cry detector tester (mic or wav)")
    ap.add_argument("--list-devices", action="store_true", help="List input audio devices and exit")
    ap.add_argument("--device", type=str, default=None, help="Device index or substring of device name")
    ap.add_argument("--wav", type=str, default=None, help="Optional WAV path. If set, test runs on this file instead of mic.")

    ap.add_argument("--tflite", type=str, default=None, help="Override TFLite path (else uses src/assets/model.tflite)")
    ap.add_argument("--classes", type=str, default=None, help="Override classes.npy path (else uses src/assets/classes.npy)")

    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--segment-sec", type=float, default=2.0)
    ap.add_argument("--hop-ratio", type=float, default=0.5)
    ap.add_argument("--cry-thr", type=float, default=0.5)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--rms-gate", type=float, default=0.001)
    ap.add_argument("--smooth-n", type=int, default=3)
    ap.add_argument("--no-bandpass", action="store_true", help="Disable band-pass prefilter")
    ap.add_argument("--print-probs", action="store_true", help="Print raw probs inside detector")

    ap.add_argument("--limit-sec", type=float, default=0.0, help="Stop after N seconds (0 = run until Ctrl+C)")
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.wav:
        run_from_wav(args)
    else:
        try:
            if args.device is not None:
                int(args.device)
        except ValueError:
            pass
        run_from_mic(args)

if __name__ == "__main__":
    main()
