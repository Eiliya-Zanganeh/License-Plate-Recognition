import sqlite3
from config import *


class Database:
    def __init__(self):
        conct = sqlite3.connect(DATABASE_PATH)
        cur = conct.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS plates (ID INTEGER PRIMARY KEY, plate text)")
        conct.commit()
        conct.close()

    def load_plates(self):
        conct = sqlite3.connect(DATABASE_PATH)
        cur = conct.cursor()
        cur.execute("SELECT * FROM plates")
        rows = cur.fetchall()
        conct.close()
        return rows

    def insert_plate(self, new_plate):
        conct = sqlite3.connect(DATABASE_PATH)
        cur = conct.cursor()
        cur.execute("INSERT INTO plates VALUES (NULL, ?) ", (new_plate,))
        conct.commit()
        conct.close()

    def delete_plate(self, plate):
        conct = sqlite3.connect(DATABASE_PATH)
        cur = conct.cursor()
        cur.execute("DELETE FROM plates WHERE plate=?", (plate,))
        conct.commit()
        conct.close()
