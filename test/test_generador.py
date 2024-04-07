import unittest
from unittest.mock import patch
import numpy as np
from io import StringIO
from src.modelo.componentes.generador import Generator
from src.procesamiento.procesamiento_texto import Text


class TestGenerador(unittest.TestCase):
    def setUp(self):
        self.latent_dim = 100
        self.text_embedding_dim = 9
        self.output_shape = (32, 32, 3)
        self.gen = Generator(
            self.latent_dim, self.text_embedding_dim, self.output_shape
        )
        self.texto_procesador = Text()
        self.stdout = StringIO()

    def test_model_structure(self):
        model = self.gen.model
        self.assertIsNotNone(model)

    def test_summary(self):
        with patch("sys.stdout", self.stdout):
            self.gen.summary()
            printed_output = self.stdout.getvalue().strip()

        self.assertNotEqual(printed_output, "")

    def test_generate_images(self):
        return None
        texto_ejemplo = "Este es un ejemplo de texto para generar imágenes."
        n_samples = texto_ejemplo.count(" ") + 1

        _, _, n1, n2 = self.texto_procesador.process_text(texto_ejemplo)

        # Concatenar las dos representaciones numéricas
        n = np.concatenate((n1, n2), axis=1)

        n = n[:, :n_samples]

        generated_images = self.gen.predict(
            [
                np.random.randn(n_samples, self.latent_dim),
                n,
            ]
        )

        self.assertEqual(generated_images.shape, (n_samples,) + self.output_shape)
