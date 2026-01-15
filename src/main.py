import time
from sensors.dht22 import read_dht22
from sensors.mlx90614 import MLX90614
from sensors.camera_supine_prone import SupineProne
from sensors.cry_detector import CryDetector




def main():
mlx = MLX90614()
cam = SupineProne()
cry = CryDetector(); cry.start()


print("[INFO] Running. Press Ctrl+C to stop.")
try:
last_pose = None
while True:
env_temp, env_hum = read_dht22()


baby_temp = None
ambient = None
try:
baby_temp = round(mlx.read_object(), 2)
ambient = round(mlx.read_ambient(), 2)
except Exception as e:
print(f"[WARN] MLX90614: {e}")


pose = cam.read_position()
if pose:
last_pose = pose


cry_state = cry.read_state()


payload = {
"environmentTemperature": env_temp,
"environmentHumidity": env_hum,
"babyTemperature": baby_temp,
"ambientTemperature": ambient,
"sleepPosition": last_pose or "unknown",
}
if cry_state:
payload.update({
"isCrying": cry_state["isCrying"],
"pCry": cry_state["pCry"],
"pNot": cry_state["pNot"],
"cryAccumulated": cry_state["accumulated"],
})


print(payload)
time.sleep(0.5)


except KeyboardInterrupt:
pass
finally:
cry.stop(); cam.close()




if __name__ == "__main__":
main()