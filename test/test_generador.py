import unittest
from unittest.mock import patch
import numpy as np
from src.gan.generador import Generator
from src.procesamiento.procesamiento_texto import Texto
from io import StringIO


class TestGenerador(unittest.TestCase):
    def setUp(self):
        self.latent_dim = 100
        self.text_embedding_dim = 50
        self.output_shape = (32, 32, 3)
        self.gen = Generator(
            self.latent_dim, self.text_embedding_dim, self.output_shape
        )
        self.texto_procesador = Texto()
        self.stdout = StringIO()

    def test_model_structure(self):
        model = self.gen.model
        self.assertIsNotNone(model)

    def test_generate_images(self):
        n_samples = 5
        texto_ejemplo = "Este es un ejemplo de texto para generar imágenes."
        self.texto_procesador.procesar_texto(texto_ejemplo)
        generated_images = self.gen.predict(
            [
                np.random.randn(n_samples, self.latent_dim),
                np.random.randn(n_samples, self.text_embedding_dim),
            ]
        )
        self.assertEqual(generated_images.shape, (n_samples,) + self.output_shape)

    def test_training_stability(self):
        # Entrena el generador durante varias épocas y verifica la estabilidad del entrenamiento
        # Puedes utilizar datos de entrenamiento simulados o un conjunto de datos real
        # Verifica la pérdida generativa y la calidad de las imágenes generadas en cada época
        return None

    def test_summary(self):
        with patch("sys.stdout", self.stdout):
            self.gen.summary()
            printed_output = self.stdout.getvalue().strip()

        self.assertNotEqual(printed_output, "")
