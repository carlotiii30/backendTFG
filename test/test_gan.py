import unittest
from src.modelo.gan import GAN
from src.modelo.componentes.discriminador import Discriminator
from src.modelo.componentes.generador import Generator


class TestGan(unittest.TestCase):
    def setUp(self):
        self.discriminator = Discriminator((32, 32, 3))
        self.generator = Generator(100, 50, (32, 32, 3))
        self.gan = GAN(self.discriminator.model, self.generator.model)

    def test_gan_structure(self):
        model = self.gan
        self.assertIsNotNone(model)
