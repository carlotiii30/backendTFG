import unittest
import numpy as np
from src.modelo.componentes.discriminador import Discriminator


class TestDiscriminator(unittest.TestCase):
    def setUp(self):
        self.input_shape = (32, 32, 3)
        self.text_embedding_dim = 9
        self.discriminator = Discriminator(self.input_shape, self.text_embedding_dim)

    def test_build_model(self):
        self.assertIsNotNone(self.discriminator.model)

    def test_summary(self):
        summary = self.discriminator.summary()
        self.assertIsInstance(summary, str)

    def test_evaluate(self):
        x = np.random.rand(10, 32, 32, 3)
        y = np.random.randint(0, 2, size=(10, 1))
        result = self.discriminator.evaluate(x, y)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_trainable(self):
        self.discriminator.trainable(True)
        self.assertTrue(self.discriminator.model.trainable)
        self.discriminator.trainable(False)
        self.assertFalse(self.discriminator.model.trainable)
