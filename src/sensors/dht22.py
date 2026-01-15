import Adafruit_DHT
from ..config import Pins


SENSOR = Adafruit_DHT.DHT22


def read_dht22(max_retries: int = 3):
"""Return (temperature_c, humidity) or (None, None) on failure."""
for _ in range(max_retries):
hum, temp = Adafruit_DHT.read_retry(SENSOR, Pins.DHT22_GPIO)
if hum is not None and temp is not None:
return float(temp), float(hum)
return None, None