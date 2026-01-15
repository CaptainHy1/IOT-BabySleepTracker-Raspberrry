import cv2
import mediapipe as mp
from picamera2 import Picamera2
from ..config import Camera as CamCfg


mp_fd = mp.solutions.face_detection


class SupineProne:
def __init__(self):
self.picam = Picamera2()
self.picam.configure(self.picam.create_preview_configuration(main={"size": (CamCfg.WIDTH, CamCfg.HEIGHT)}))
self.picam.start()
self.detector = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)
self._frame_skip = 0


def read_position(self):
"""Return 'supine' if a face is detected, otherwise 'prone'.
To save CPU, we only infer every N frames.
"""
frame = self.picam.capture_array()
if self._frame_skip % CamCfg.INFER_EVERY_N_FRAMES != 0:
self._frame_skip += 1
return None
self._frame_skip += 1
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
res = self.detector.process(rgb)
return "supine" if res.detections else "prone"


def close(self):
self.picam.stop()
self.detector.close()