from keras.layers import Conv2D, Flatten, Dropout, LeakyReLU, Dense
from keras.models import Sequential
from keras.optimizers import Adam


class Discriminator:
    """Clase que define el discriminador de una Red Generativa Adversaria (GAN).

    Esta clase representa el discriminador de una GAN, que se encarga de
    discriminar entre imágenes reales y generadas por el generador.

    Attributes:
        input_shape (tuple): Forma de las imágenes de entrada al discriminador.
        model (keras.models.Sequential): Modelo del discriminador.
    """

    def __init__(self, input_shape):
        """Inicializa el discriminador con la forma de entrada especificada.

        Args:
            input_shape (tuple): Forma de las imágenes de entrada al
            discriminador.
        """
        self.input_shape = input_shape
        self.model = self.build_model()

    def build_model(self):
        """Construye y compila el modelo del discriminador.

        Returns:
            keras.models.Sequential: Modelo del discriminador.
        """
        model = Sequential()

        # Capa convolucional
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(LeakyReLU(negative_slope=0.2))

        # Capa convolucional
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(negative_slope=0.2))

        # Capa convolucional
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(negative_slope=0.2))

        # Capa convolucional
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(negative_slope=0.2))

        # Capa de aplanamiento
        model.add(Flatten())

        # Capa de dropout
        model.add(Dropout(0.4))

        # Capa densa
        model.add(Dense(1, activation="sigmoid"))

        # Optimizador
        optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

        # Compilar el modelo
        model.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        return model

    def summary(self):
        """Imprime un resumen del modelo del discriminador."""
        return self.model.summary()

    def evaluate(self, x, y):
        """Evalúa el modelo del discriminador en un conjunto de datos de
        entrada y etiquetas.

        Args:
            x (numpy.ndarray): Conjunto de datos de entrada.
            y (numpy.ndarray): Etiquetas verdaderas correspondientes a los datos
            de entrada.

        Returns:
            list: Lista que contiene la pérdida y la precisión del modelo en el
              conjunto de datos de entrada.
        """
        return self.model.evaluate(x, y)

    def trainable(self, trainable):
        """Establece si el modelo del discriminador es entrenable o no.

        Args:
            trainable (bool): Indica si el modelo del discriminador es
            entrenable o no.
        """
        self.model.trainable = trainable
