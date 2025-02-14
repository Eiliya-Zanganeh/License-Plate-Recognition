import cv2 as cv

from Detector_Module.detector import Detector
from Recogniser_Module.recogniser import Recogniser
from config import *


class Identification:
    def __init__(self):
        self.detector = Detector()
        self.recogniser = Recogniser()

    def __call__(self, img_rgb):
        img_rgb, plates = self.detector(img_rgb)
        outputs = []
        if len(plates) > 0:
            for plate in plates:
                image = plate['image']
                x1, y1, x2, y2 = plate['box']
                predict = self.recogniser(image)
                if predict[1] > RECOGNISER_THRESHOLD:
                    cv.putText(img_rgb, str(predict[0]), (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    outputs.append(predict)
        return img_rgb, outputs
