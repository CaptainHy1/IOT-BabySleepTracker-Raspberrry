# Baby Monitor Pi

## Setup

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

## Default pins

- DHT22 DATA: GPIO4 (BOARD pin 7)
- I2C (MLX90614): SDA = BOARD pin 3, SCL = BOARD pin 5
- Camera: Pi Camera via Picamera2

IMPORTANT: Raspberry Pi BOARD pin 9 is GND. It cannot be used as DHT22 DATA.

## Run

bash run.sh
