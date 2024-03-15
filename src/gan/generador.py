from keras.layers import Dense, Reshape, Conv2DTranspose, LeakyReLU, Conv2D, Input
from keras.models import Sequential, Model

# Clase que define el generador de la GAN
class Generador:
    def __init__(self, latent_dim, output_shape):
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # Capa de entrada
        model.add(Input(shape=(self.latent_dim,)))

        # Capa densa
        n_nodes = 256 * 4 * 4
        model.add(Dense(n_nodes))
        model.add(LeakyReLU())
        model.add(Reshape((4, 4, 256)))

        # Capas de convolución transpuesta
        model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(negative_slope=0.2))

        model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(negative_slope=0.2))

        model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(negative_slope=0.2))

        # Capa de salida
        model.add(Conv2D(3, kernel_size=3, activation="tanh", padding="same"))

        return model

    def summary(self):
        return self.model.summary()

    def predict(self, x_input):
        return self.model.predict(x_input)