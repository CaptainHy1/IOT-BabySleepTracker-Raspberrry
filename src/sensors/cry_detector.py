import time
import queue
from collections import deque
import numpy as np
import sounddevice as sd
import librosa

# Try TensorFlow first (tf.lite), then fall back to tflite-runtime
def _load_interpreter(model_path: str):
    try:
        import tensorflow as tf
        if hasattr(tf, "lite"):
            inter = tf.lite.Interpreter(model_path=model_path)
            inter.allocate_tensors()
            return inter
    except Exception:
        pass
    try:
        import tflite_runtime.interpreter as tflite
        inter = tflite.Interpreter(model_path=model_path)
        inter.allocate_tensors()
        return inter
    except Exception as e:
        raise RuntimeError(f"Cannot create TFLite interpreter: {e}")

# Optional band-pass 300..4000 Hz
def _bandpass(x: np.ndarray, sr: int, low=300, high=4000):
    try:
        from scipy.signal import butter, lfilter
        nyq = 0.5 * sr
        b, a = butter(4, [low/nyq, high/nyq], btype="band")
        return lfilter(b, a, x).astype(np.float32)
    except Exception:
        return x

def _mel_image(
    audio: np.ndarray,
    sr: int = 16000,
    target_hw=(128, 128),
    n_fft=1024,
    hop_length=256,
    n_mels=128,
    use_bandpass=True,
):
    # Optional denoise band
    x = _bandpass(audio, sr) if use_bandpass else audio
    # Mel power spectrogram
    S = librosa.feature.melspectrogram(
        y=x, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0
    )
    # Fixed dB scale, then clip to [-80, 0]
    logmel = librosa.power_to_db(S, ref=1.0)
    logmel = np.clip(logmel, -80.0, 0.0)
    # Map to [0,1]
    img01 = (logmel + 80.0) / 80.0
    # Resize to target_hw using Pillow (keep simple, LANCZOS)
    try:
        from PIL import Image
        img_u8 = (img01 * 255.0).astype(np.uint8)
        pil = Image.fromarray(img_u8)
        pil = pil.resize(target_hw, Image.Resampling.LANCZOS)
        arr = np.array(pil).astype(np.float32) / 255.0
    except Exception:
        # Lightweight fallback: pad/crop then simple resample by slicing
        # (not as good as PIL, but keeps shapes right)
        h, w = img01.shape
        th, tw = target_hw
        # pad/crop width
        if w < tw:
            pad = tw - w
            img01 = np.pad(img01, ((0, 0), (0, pad)))
        arr = img01[:, :tw]
        if h != th:
            # naive vertical resample
            ys = np.linspace(0, h - 1, th).astype(np.int32)
            arr = arr[ys, :]
    # Stack to 3 channels (RGB-like)
    rgb = np.stack([arr] * 3, axis=-1).astype(np.float32)
    return rgb

class _MicStream:
    """Overlap audio windowing: segment_sec with hop_ratio."""
    def __init__(self, sr=16000, segment_sec=2.0, hop_ratio=0.5, device=None):
        self.sr = sr
        self.segment_len = int(segment_sec * sr)
        self.hop_len = max(1, int(self.segment_len * hop_ratio))
        self.device = device
        self.q = queue.Queue()
        self.buf = np.zeros(0, dtype=np.float32)
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        mono = indata.reshape(-1).astype(np.float32)
        self.q.put(mono)

    def start(self):
        self.stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
            blocksize=0,
        )
        self.stream.start()

    def next_segment(self):
        # Pull whatever is in the queue
        pulled = False
        while True:
            try:
                chunk = self.q.get_nowait()
                self.buf = np.concatenate([self.buf, chunk])
                pulled = True
            except queue.Empty:
                break
        if not pulled:
            return None
        if len(self.buf) >= self.segment_len:
            seg = self.buf[:self.segment_len].copy()
            self.buf = self.buf[self.hop_len:]
            return seg
        return None

    def stop(self):
        try:
            if self.stream:
                self.stream.stop(); self.stream.close()
        except Exception:
            pass

class CryDetector:
    """
    Matches the behavior you described:
    - Mel log dB in [-80, 0], resized to 128x128, 3 channels
    - Optional band-pass 300..4000 Hz
    - RMS gate to skip quiet frames
    - Moving-average smoothing over N frames
    - Min-duration requirement before raising 'isCrying'
    - Works with float or int8 quantized TFLite models
    - Aligns with classes.npy for 'cry' / 'not_cry'
    """

    def __init__(
        self,
        tflite_path="src/assets/modef.tflite",
        classes_path="src/assets/classes.npy",
        sr=16000,
        segment_sec=2.0,
        hop_ratio=0.5,
        cry_threshold=0.6,
        min_duration=2.0,
        rms_gate=0.005,
        smooth_n=5,
        use_bandpass=True,
        print_probs=False,
        device=None,
    ):
        # Audio
        self.sr = sr
        self.segment_sec = segment_sec
        self.hop_ratio = hop_ratio
        self.frame_hop_s = segment_sec * hop_ratio
        self.rms_gate = float(rms_gate)
        self.use_bandpass = bool(use_bandpass)

        # Smoothing / decision
        self.cry_threshold = float(cry_threshold)
        self.min_duration = float(min_duration)
        self.above_streak_s = 0.0
        self.hist = deque(maxlen=max(1, int(smooth_n)))
        self.print_probs = bool(print_probs)

        # Model
        self.interp = _load_interpreter(tflite_path)
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.in_is_int8 = (self.inp["dtype"] == np.int8)
        self.out_is_int8 = (self.out["dtype"] == np.int8)
        self.in_q = self.inp.get("quantization", (0.0, 0))
        self.out_q = self.out.get("quantization", (0.0, 0))

        # Classes
        self.classes = [str(c) for c in np.load(classes_path, allow_pickle=True)]
        names_lower = [c.lower() for c in self.classes]

        def _find(*cands):
            for c in cands:
                if c in names_lower:
                    return names_lower.index(c)
            return None

        self.cry_idx = _find("cry")
        self.not_idx = _find("not_cry", "no_cry", "non_cry", "notcry", "nocry")

        # Mic
        self.mic = _MicStream(sr=sr, segment_sec=segment_sec, hop_ratio=hop_ratio, device=device)
        self.mic.start()

    def stop(self):
        self.mic.stop()

    def _infer_probs(self, img):
        x = np.expand_dims(img, axis=0).astype(np.float32)
        if self.in_is_int8:
            s, z = self.in_q
            s = s if s != 0 else 1.0
            x = (x / s + z).astype(np.int8)
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        out = self.interp.get_tensor(self.out["index"])
        if self.out_is_int8:
            s, z = self.out_q
            out = (out.astype(np.float32) - z) * (s if s != 0 else 1.0)
        # softmax
        e = np.exp(out[0] - np.max(out[0]))
        probs = e / np.sum(e)
        return probs.astype(float)

    def read_state(self):
        """Return a dict when a new window is processed, else None."""
        seg = self.mic.next_segment()
        if seg is None:
            return None

        # RMS gate: very quiet -> treat as not crying
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms < self.rms_gate:
            pcry, pnot = 0.0, 1.0
            self.hist.append(pcry)
            ps = float(np.mean(self.hist))
            isCrying = ps >= self.cry_threshold
            above = False
        else:
            img = _mel_image(
                seg,
                sr=self.sr,
                target_hw=(128, 128),
                n_fft=1024,
                hop_length=256,
                n_mels=128,
                use_bandpass=self.use_bandpass,
            )
            probs = self._infer_probs(img)

            # map to pCry / pNot
            if self.cry_idx is not None and self.not_idx is not None:
                pcry = float(probs[self.cry_idx])
                pnot = float(probs[self.not_idx])
            elif probs.size == 2:
                if self.cry_idx is not None:
                    pcry = float(probs[self.cry_idx])
                    pnot = 1.0 - pcry
                else:
                    # assume index 1 is 'cry'
                    pcry = float(probs[1]); pnot = float(probs[0])
            elif self.cry_idx is not None:
                pcry = float(probs[self.cry_idx]); pnot = 1.0 - pcry
            else:
                m = int(np.argmax(probs))
                pcry = float(probs[m]); pnot = 1.0 - pcry

            self.hist.append(pcry)
            ps = float(np.mean(self.hist))
            isCrying = ps >= self.cry_threshold
            above = isCrying

            if self.print_probs:
                # optional verbose output (do not print here; main() can do it)
                pass

        # min-duration logic using hop seconds
        if above:
            self.above_streak_s += self.frame_hop_s
        else:
            self.above_streak_s = max(0.0, self.above_streak_s - self.frame_hop_s)

        event = (self.above_streak_s >= self.min_duration)
        if event:
            # reset so next continuous exceedance triggers again
            self.above_streak_s = 0.0

        return {
            "isCrying": isCrying,
            "pCry": round(pcry, 4),
            "pNot": round(pnot, 4),
            "smooth": round(ps, 4),
            "rms": round(rms, 6),
            "event": event,  # True right when min-duration satisfied
        }
