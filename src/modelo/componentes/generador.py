from keras.layers import (
    Dense,
    Reshape,
    Conv2DTranspose,
    LeakyReLU,
    Conv2D,
    Concatenate,
    Input,
)
from keras.models import Model


class Generator:
    """Clase que define el generador de una Red Generativa Adversaria
    Condicional(cGAN).

    Esta clase representa el generador de una GAN, que se encarga de generar
    imágenes a partir de un vector de ruido de dimensión latente y de un texto
    que condiciona la generación de imágenes.

    Attributes:
        latent_dim (int): Dimensión del espacio latente.
        text_embedding_dim (int): Dimensión del espacio de incrustación de texto.
        output_shape (tuple): Forma de la salida del generador.
        model (keras.models.Sequential): Modelo del generador.
    """

    def __init__(self, latent_dim, text_embedding_dim, output_shape):
        """Inicializa el generador con la dimensión latente y la forma de
        salida especificadas.

        Args:
            latent_dim (int): Dimensión del espacio latente.
            text_embedding_dim (int): Dimensión del espacio de incrustación de texto.
            output_shape (tuple): Forma de la salida del generador.
        """
        self.latent_dim = latent_dim
        self.text_embedding_dim = text_embedding_dim
        self.output_shape = output_shape
        self.model = self.build_model()

    def build_model(self):
        """Construye y compila el modelo del generador.

        Returns:
            keras.models.Sequential: Modelo del generador.
        """

        # Entradas
        input_latent = Input(shape=(self.latent_dim,))
        input_text = Input(shape=(self.text_embedding_dim,))

        # Capa densa para combinar el ruido y el texto
        combined_input = Concatenate()([input_latent, input_text])
        x = Dense(256 * 4 * 4)(combined_input)
        x = LeakyReLU()(x)
        x = Reshape((4, 4, 256))(x)

        # Capas de convolución transpuesta
        x = Conv2DTranspose(128, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU(negative_slope=0.2)(x)

        x = Conv2DTranspose(128, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU(negative_slope=0.2)(x)

        x = Conv2DTranspose(128, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU(negative_slope=0.2)(x)

        # Capa de salida
        output = Conv2D(3, kernel_size=3, activation="tanh", padding="same")(x)

        # Modelo
        model = Model(inputs=[input_latent, input_text], outputs=output)

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

    def save(self, filename):
        """Guarda el modelo del generador en un archivo.

        Args:
            filename (str): Nombre del archivo donde se guardará el modelo.
        """
        self.model.save(filename)
