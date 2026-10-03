import smbus2
from ..config import I2C, Calib

OBJ1 = 0x07
TA   = 0x06

class MLX90614:
    def __init__(self, bus: int = I2C.BUS, addr: int = I2C.MLX90614_ADDR):
        self.bus = smbus2.SMBus(bus)
        self.addr = addr

    def _read_word(self, reg):
        raw = self.bus.read_word_data(self.addr, reg)  # no byte swap
        return raw

    def read_object(self):
        # convert to C then add calibration offset
        return (self._word_to_celsius(self._read_word(OBJ1)) + Calib.MLX_OBJ_OFFSET_C)

    def read_ambient(self):
        return (self._word_to_celsius(self._read_word(TA)) + Calib.MLX_AMB_OFFSET_C)

    @staticmethod
    def _word_to_celsius(raw):
        return raw * 0.02 - 273.15
