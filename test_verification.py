from License_Plate_Recognition_Module.verification import Verification
import cv2 as cv

verification = Verification()
print(verification.get_plates())
image = cv.imread('images/img_1.png')
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
print(f'Is verification successful? {verification(image)}')
verification.insert_plate('54l55455')
print(verification.get_plates())
print(f'Is verification successful? {verification(image)}')
