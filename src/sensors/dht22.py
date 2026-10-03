import time
import board
import adafruit_dht

# DATA on GPIO4 (BOARD pin 7). Change board.D4 if you wire another pin.
_dht = adafruit_dht.DHT22(board.D17, use_pulseio=False)

def read_dht22(max_retries: int = 3):
    """Return (temperature_c, humidity) or (None, None) on failure."""
    for _ in range(max_retries):
        try:
            t = _dht.temperature
            h = _dht.humidity
            if (t is not None) and (h is not None):
                return float(t), float(h)
        except RuntimeError:
            # common transient read errors; wait a bit and retry
            time.sleep(0.2)
        except Exception:
            # sensor not found or GPIO issue
            break
    return None, None
