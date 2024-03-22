import unittest
from unittest.mock import patch
from io import StringIO
from keras.datasets import cifar100
from src.gan.discriminador import Discriminator
from src.prueba.entrenamiento import Training


class TestDiscriminador(unittest.TestCase):
    def setUp(self):
        self.input_shape = (32, 32, 3)
        self.discriminador = Discriminator(self.input_shape)
        self.dataset = Training.load_images(cifar100)
        self.stdout = StringIO()

    def test_model_structure(self):
        model = self.discriminador.model
        self.assertIsNotNone(model)

    def test_real_training(self):
        self.setUp()
        dataset, labels = Training.load_real_data(self.dataset, 100)
        initial_loss, initial_accuracy = self.discriminador.evaluate(dataset, labels)
        Training.train_discriminator(self.discriminador, self.dataset)
        loss, accuracy = self.discriminador.evaluate(dataset, labels)
        self.assertTrue(accuracy > initial_accuracy)

    def test_fake_training(self):
        self.setUp()
        dataset, labels = Training.load_fake_data(100)
        initial_loss, initial_accuracy = self.discriminador.evaluate(dataset, labels)
        Training.train_discriminator(self.discriminador, self.dataset)
        loss, accuracy = self.discriminador.evaluate(dataset, labels)
        self.assertTrue(accuracy > initial_accuracy)

    def test_build_model(self):
        model = self.discriminador.build_model()
        self.assertIsNotNone(model)

    def test_summary(self):
        with patch("sys.stdout", self.stdout):
            self.discriminador.summary()
            printed_output = self.stdout.getvalue().strip()

        self.assertNotEqual(printed_output, "")

    def test_evaluate(self):
        dataset, labels = Training.load_real_data(self.dataset, 100)
        result = self.discriminador.evaluate(dataset, labels)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_trainable(self):
        trainable = self.discriminador.trainable
        self.assertIsNotNone(trainable)
