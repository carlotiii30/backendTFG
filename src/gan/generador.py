# Generador de la GAN

from keras.layers import Dense, Reshape, Conv2DTranspose, LeakyReLU, Conv2D
from keras.models import Sequential


# Clase del generador
class Generador:
    def __init__(self, latent_dim, output_shape):
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # Capa densa
        n_nodes = 256 * 4 * 4
        model.add(Dense(n_nodes, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((4, 4, 256)))

        # Capa de convolución transpuesta
        model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        # Capa de convolución transpuesta
        model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        # Capa de convolución transpuesta
        model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        # Capa de salida
        model.add(Conv2D(3, kernel_size=3, activation="tanh", padding="same"))

        return model

    def summary(self):
        return self.model.summary()
