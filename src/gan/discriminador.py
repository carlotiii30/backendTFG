# Discriminador de la GAN

from keras.layers import Conv2D, Flatten, Dropout, LeakyReLU, Dense
from keras.models import Sequential
from keras.optimizers import Adam

# Clase del discriminador
class Discriminator:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # Capa convolucional
        model.add(Conv2D(64, kernel_size=3, input_shape=self.input_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        # Capa convolucional
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        # Capa convolucional
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        # Capa convolucional
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        # Capa de aplanamiento
        model.add(Flatten())

        # Capa de dropout
        model.add(Dropout(0.4))

        # Capa densa
        model.add(Dense(1, activation="sigmoid"))

        # Optimizador
        optimizer = Adam(lr=0.0002, beta_1=0.5)

        # Compilar el modelo
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        return model

    def summary(self):
        return self.model.summary()