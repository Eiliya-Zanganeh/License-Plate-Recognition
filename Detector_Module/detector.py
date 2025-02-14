from ultralytics import YOLO
import numpy as np
import cv2 as cv

from config import *


class Detector:
    def __init__(self):
        self.model = YOLO(DETECTOR_WEIGHT)

    def __call__(self, image_rgb: np.ndarray):
        result = self.model(image_rgb)[0]
        plates = []
        for box in result.boxes:
            inference = box.conf[0]
            if inference > DETECTOR_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
                cropped_image = image_rgb[y1:y2, x1:x2]
                new_plate = {
                    "image": cropped_image,
                    "inference": inference,
                    "box": [x1, y1, x2, y2]
                }
                plates.append(new_plate)
        return image_rgb, plates
