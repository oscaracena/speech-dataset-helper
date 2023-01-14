
import logging
logging.getLogger("faker").setLevel(logging.ERROR)

from unittest import TestCase
from faker import Factory

from sdh import DatasetProject

fake = Factory.create("es_ES")


class DatasetProjectTests(TestCase):
    def _new_sentence(self, content=None, path=None):
        return {
            "content": content or fake.text(100),
            "clip-path": path or fake.file_name(extension="wav"),
        }

    def _check_sentences(self, sentences):
        prj = DatasetProject("")
        for source, expected in sentences.items():
            sentence = self._new_sentence(source)
            result = prj._sentence_to_meta(sentence).split("|")[-1].strip()
            self.assertEqual(result, expected)

    def test_normalize_basic_numbers(self):
        self._check_sentences({
            "prueba número 12":
            "prueba número doce",

            "En 1625, Francia e Inglaterra":
            "en mil seiscientos veinticinco, francia e inglaterra",

            "En total, 11.":
            "en total, once.",

            "Los 3 mosqueteros (1844) y El Conde de Montecristo (1849),":
            "los tres mosqueteros (mil ochocientos cuarenta y cuatro) y el conde de montecristo "
            "(mil ochocientos cuarenta y nueve),",

            # "Ejecute el comando G45L":
            # "Ejecute el comando ge cuarenta y cinco ele"
        })

    def test_roman_numbers(self):
        self._check_sentences({
            "se dice siglo I hasta siglo X, luego es siglo XI, etc.":
            "se dice siglo primero hasta siglo décimo, luego es siglo once, etc.",

            "en boga durante el siglo XVII.":
            "en boga durante el siglo diecisiete.",

            # "MMDCCLXXIII":
            # "dos mil setecientos setenta y tres",

            # "duque Victorio Amadeo II,":
            # "duque victorio amadeo segundo,",
        })

    def test_abreviations(self):
        self._check_sentences({
            # "Algo llamado Dionisio, o el D. Paco":
            # "algo llamado dionisio, o el don paco",
        })

    def test_roman_to_number(self):
        prj = DatasetProject("")
        numbers = {
            "C": 100, "M": 1000, "XII": 12, "II": 2,
            "MDLXXV": 1575,
            "MMDCCLXXIII": 2773,
        }

        for source, expected in numbers.items():
            result = prj._roman_to_number(source)
            self.assertEquals(result, expected)

    def test_roman_to_number_fails(self):
        prj = DatasetProject("")
        others = ["nonum", 134, "124", None, "seis"]

        for item in others:
            result = prj._roman_to_number(item)
            self.assertEquals(result, None)
