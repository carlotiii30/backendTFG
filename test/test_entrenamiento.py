import unittest
import numpy as np
from unittest.mock import patch
from keras.datasets import cifar10
from src.procesamiento.entrenamiento import Training
from src.modelo.componentes.discriminador import Discriminator
from src.modelo.componentes.generador import Generator


class TestEntrenamiento(unittest.TestCase):
    def setUp(self):
        self.dataset = cifar10
        self.generator = Generator(100, 9, (32, 32, 3))
        self.discriminator = Discriminator((32, 32, 3), 9)

    def test_load_images(self):
        X = Training.load_images(self.dataset)
        self.assertEqual(X.shape, (50000, 32, 32, 3))

    def test_load_real_data(self):
        n_samples = 10
        processed_data = Training.load_images(self.dataset)
        X, y = Training.load_real_data(processed_data, n_samples)
        self.assertEqual(X.shape, (n_samples, 32, 32, 3))
        self.assertEqual(y.shape, (n_samples, 1))

    def test_generate_fake_data(self):
        condition = np.random.rand(1, 9)
        n_samples = 10
        images, y = Training.generate_fake_data(self.generator, condition, n_samples)
        self.assertEqual(images.shape, (n_samples, 32, 32, 3))
        self.assertEqual(y.shape, (n_samples, 1))

    def test_train_step(self):
        X = np.random.rand(10, 32, 32, 3)
        y = np.random.randint(0, 2, (10, 1))
        condition = np.random.rand(10, 9)
        acc = Training.train_step(self.discriminator.model, [X, condition], y)
        self.assertIsInstance(float(acc), float)
