import unittest

import numpy as np
from src.procesamiento.procesamiento_texto import Texto


class TestProcesamientoTexto(unittest.TestCase):
    def test_procesar_texto(self):
        texto = "Hello, world! This is a sample text."
        expected_texto = "hello world this is a sample text"
        expected_tokens = ["hello", "world", "this", "is", "a", "sample", "text"]
        expected_representacion_numerica = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )

        result_texto, result_tokens, result_representacion_numerica = (
            Texto.procesar_texto(texto)
        )

        self.assertEqual(result_texto, expected_texto)
        self.assertEqual(result_tokens, expected_tokens)
        self.assertTrue(
            np.array_equal(
                result_representacion_numerica, expected_representacion_numerica
            )
        )
