from dataclasses import dataclass


@dataclass
class Pins:
# Adafruit_DHT expects BCM GPIO number, not BOARD number
DHT22_GPIO: int = 4 # equals BOARD pin 7. Pin 9 is GND and cannot be used.


@dataclass
class I2C:
BUS: int = 1
MLX90614_ADDR: int = 0x5A


@dataclass
class Audio:
SAMPLE_RATE: int = 16000
BLOCK_SECONDS: float = 1.0 # inference window per block
MEL_SIZE: int = 128


@dataclass
class Camera:
WIDTH: int = 640
HEIGHT: int = 480
FPS: int = 8
INFER_EVERY_N_FRAMES: int = 2


@dataclass
class CryThreshold:
PROB: float = 0.6 # pCry >= threshold -> isCrying = True
MIN_SECONDS: float = 2.0 # require continuous time above threshold


class Paths:
TFLITE = "src/assets/modef.tflite"
CLASSES = "src/assets/classes.npy"