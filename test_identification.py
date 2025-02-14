from License_Plate_Recognition_Module.identification import Identification
import cv2 as cv

image = cv.imread('images/img_1.png')
recognizer = Identification()
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image, plates = recognizer(image)
print('===========================')
for plate in plates:
    print(plate)
image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
cv.imshow('image', image)
cv.waitKey(0)
