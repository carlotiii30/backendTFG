import unittest
from keras.datasets import cifar100
from src.gan.discriminador import Discriminator
from src.prueba.entrenamiento import Training


class TestDiscriminador(unittest.TestCase):
    def setUp(self):
        self.input_shape = (32, 32, 3)
        self.discriminador = Discriminator(self.input_shape)
        self.dataset = Training.load_images(cifar100)

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


if __name__ == "__main__":
    unittest.main()
