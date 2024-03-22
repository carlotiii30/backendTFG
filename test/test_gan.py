import unittest
from src.prueba.gan import GAN
from src.gan.discriminador import Discriminator
from src.gan.generador import Generator


class TestGan(unittest.TestCase):
    def setUp(self):
        self.discriminator = Discriminator((32, 32, 3))
        self.generator = Generator(100, (32, 32, 3))
        self.gan = GAN(self.discriminator.model, self.generator.model)

    def test_gan_structure(self):
        model = self.gan
        self.assertIsNotNone(model)
