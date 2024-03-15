from keras.models import Sequential
from keras.optimizers import Adam
from keras.datasets import cifar100
from src.gan.discriminador import Discriminator
from src.gan.generador import Generator
from src.prueba.entrenamiento import Training


def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    gan.compile(loss='binary_crossentropy', optimizer=opt)

    return gan

dataset = Training.load_images(cifar100)
discriminator = Discriminator((32, 32, 3))
Training.train_discriminator(discriminator, dataset)

generator = Generator(100, (32, 32, 3))

gan = create_gan(discriminator.model, generator.model)
gan.summary()