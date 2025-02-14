from difflib import SequenceMatcher

from License_Plate_Recognition_Module.database import Database
from License_Plate_Recognition_Module.identification import Identification
from config import *


class Verification:
    def __init__(self):
        self.database = Database()
        self.plates = self.database.load_plates()
        self.identification = Identification()

    def insert_plate(self, plate):
        self.database.insert_plate(plate)
        self.plates = self.database.load_plates()

    def delete_plate(self, plate):
        self.database.delete_plate(plate)
        self.plates = self.database.load_plates()

    def get_plates(self):
        return self.plates

    def __call__(self, image_rgb):
        plates = self.identification(image_rgb)[1]
        for plate in plates:
            for row in self.plates:
                num = SequenceMatcher(None, plate[0], row[1]).ratio()
                if num > VERIFICATION_THRESHOLD:
                    return True
        return False
