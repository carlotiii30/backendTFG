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
    """Class representing a generator model for a Generative Adversarial Network (GAN).

    This class defines the generator model, which is responsible for generating
    images based on a given input, typically a latent space vector.

    Attributes:
        latent_dim (int): Dimension of the latent space.
        text_embedding_dim (int): Dimension of the text embedding.
        output_shape (tuple): Shape of the output image.
        model (keras.models.Model): Generator model.

    Methods:
        build_model(): Builds the generator model.
        summary(): Prints a summary of the generator model.
        predict(x_input): Generates an output image given an input.
        save(filename): Saves the generator model to a file.
    """

    def __init__(self, latent_dim, text_embedding_dim, output_shape):
        """Initializes the Generator object.

        Args:
            latent_dim (int): Dimension of the latent space.
            text_embedding_dim (int): Dimension of the text embedding.
            output_shape (tuple): Shape of the output image.
        """
        self.latent_dim = latent_dim
        self.text_embedding_dim = text_embedding_dim
        self.output_shape = output_shape
        self.model = self.build_model()

    def build_model(self):
        """Builds the generator model.

        Returns:
            keras.models.Model: Generator model.
        """

        # Inputs
        input_latent = Input(shape=(self.latent_dim,))
        input_text = Input(shape=(self.text_embedding_dim,))

        # Dense layer
        combined_input = Concatenate()([input_latent, input_text])
        x = Dense(256 * 4 * 4)(combined_input)
        x = LeakyReLU()(x)
        x = Reshape((4, 4, 256))(x)

        # Convolutional layers
        x = Conv2DTranspose(128, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU(negative_slope=0.2)(x)

        x = Conv2DTranspose(128, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU(negative_slope=0.2)(x)

        x = Conv2DTranspose(128, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU(negative_slope=0.2)(x)

        # Output layer
        output = Conv2D(3, kernel_size=3, activation="tanh", padding="same")(x)

        # Model
        model = Model(inputs=[input_latent, input_text], outputs=output)

        return model

    def summary(self):
        """Prints a summary of the generator model."""
        return self.model.summary()

    def predict(self, x_input):
        """Generates an output image given an input.

        Args:
            x_input: The input to the generator model.

        Returns:
            numpy.ndarray: The generated output image.
        """
        return self.model.predict(x_input)

    def save(self, filename):
        """Saves the generator model to a file.

        Args:
            filename (str): Name of the file to save the model to.
        """
        self.model.save(filename)
