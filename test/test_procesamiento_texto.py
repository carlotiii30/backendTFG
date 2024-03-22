import unittest
from src.procesamiento.procesamiento_texto import Texto

class TestProcesamientoTexto(unittest.TestCase):
    def test_procesar_texto(self):
        texto = "Hello, world! This is a sample text."
        expected_texto = "hello world this is a sample text"
        expected_tokens = ["hello", "world", "this", "is", "a", "sample", "text"]
        expected_representacion_numerica = {
            "hello": [1, 0, 0, 0, 0, 0, 0],
            "world": [0, 1, 0, 0, 0, 0, 0],
            "this": [0, 0, 1, 0, 0, 0, 0],
            "is": [0, 0, 0, 1, 0, 0, 0],
            "a": [0, 0, 0, 0, 1, 0, 0],
            "sample": [0, 0, 0, 0, 0, 1, 0],
            "text": [0, 0, 0, 0, 0, 0, 1]
        }

        result_texto, result_tokens, result_representacion_numerica = Texto.procesar_texto(texto)

        self.assertEqual(result_texto, expected_texto)
        self.assertEqual(result_tokens, expected_tokens)
        self.assertEqual(result_representacion_numerica, expected_representacion_numerica)