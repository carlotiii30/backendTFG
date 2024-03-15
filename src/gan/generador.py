from keras.layers import Dense, Reshape, Conv2DTranspose, LeakyReLU, Conv2D, Input
from keras.models import Sequential, Model


class Generator:
    """Clase que define el generador de una Red Generativa Adversaria (GAN).

    Esta clase representa el generador de una GAN, que se encarga de generar imágenes a partir de un vector de ruido
    de dimensión latente.

    Attributes:
        latent_dim (int): Dimensión del espacio latente.
        output_shape (tuple): Forma de la salida del generador.
        model (keras.models.Sequential): Modelo del generador.
    """

    def __init__(self, latent_dim, output_shape):
        """Inicializa el generador con la dimensión latente y la forma de salida especificadas.

        Args:
            latent_dim (int): Dimensión del espacio latente.
            output_shape (tuple): Forma de la salida del generador.
        """
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.model = self.build_model()

    def build_model(self):
        """Construye y compila el modelo del generador.

        Returns:
            keras.models.Sequential: Modelo del generador.
        """
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
        """Imprime un resumen del modelo del generador."""
        return self.model.summary()

    def predict(self, x_input):
        """Genera imágenes a partir de un vector de entrada.

        Args:
            x_input (numpy.ndarray): Vector de entrada.

        Returns:
            numpy.ndarray: Imágenes generadas por el generador.
        """
        return self.model.predict(x_input)
