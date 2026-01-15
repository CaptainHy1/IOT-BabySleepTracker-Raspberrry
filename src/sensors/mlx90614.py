import smbus2
from ..config import I2C


OBJ1 = 0x07
TA = 0x06


class MLX90614:
def __init__(self, bus: int = I2C.BUS, addr: int = I2C.MLX90614_ADDR):
self.bus = smbus2.SMBus(bus)
self.addr = addr


def _read_word(self, reg):
raw = self.bus.read_word_data(self.addr, reg)
# swap bytes according to MLX90614 SMBus behavior
raw = ((raw & 0xFF) << 8) | (raw >> 8)
return raw


def read_object(self):
return self._word_to_celsius(self._read_word(OBJ1))


def read_ambient(self):
return self._word_to_celsius(self._read_word(TA))


@staticmethod
def _word_to_celsius(raw):
return raw * 0.02 - 273.15