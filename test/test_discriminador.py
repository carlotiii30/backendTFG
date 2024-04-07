import unittest
from unittest.mock import patch
import numpy as np
from io import StringIO
from src.modelo.componentes.discriminador import Discriminator


class TestDiscriminator(unittest.TestCase):
    def setUp(self):
        self.input_shape = (32, 32, 3)
        self.text_embedding_dim = 9
        self.discriminator = Discriminator(self.input_shape, self.text_embedding_dim)
        self.stdout = StringIO()

    def test_build_model(self):
        self.assertIsNotNone(self.discriminator.model)

    def test_summary(self):
        with patch("sys.stdout", self.stdout):
            self.discriminator.summary()
            printed_output = self.stdout.getvalue().strip()

        self.assertNotEqual(printed_output, "")

    def test_evaluate(self):
        x_image = np.random.rand(10, 32, 32, 3)
        x_text = np.random.rand(10, self.text_embedding_dim)
        y = np.random.randint(0, 2, size=(10, 1))
        result = self.discriminator.evaluate([x_image, x_text], y)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_trainable(self):
        self.discriminator.trainable(True)
        self.assertTrue(self.discriminator.model.trainable)
        self.discriminator.trainable(False)
        self.assertFalse(self.discriminator.model.trainable)
