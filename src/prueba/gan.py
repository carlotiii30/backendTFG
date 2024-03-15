from keras.models import Sequential
from keras.optimizers import Adam
from keras.datasets import cifar100
from src.gan.discriminador import Discriminator
from src.gan.generador import Generator
from src.prueba.entrenamiento import Training


class GAN:
    def __init__(self, discriminator, generator):
        self.gan = self.create_gan(discriminator, generator)

    def create_gan(self, discriminator, generator):
        discriminator.trainable = False
        gan = Sequential()
        gan.add(generator)
        gan.add(discriminator)

        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        gan.compile(loss="binary_crossentropy", optimizer=opt)

        return gan

    def summary(self):
        self.gan.summary()