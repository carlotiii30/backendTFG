import numpy as np
from keras.layers import (
    Conv2D,
    Flatten,
    Dropout,
    LeakyReLU,
    Dense,
    Input,
    Concatenate,
    Reshape,
    RepeatVector,
)
from keras.models import Model
from keras.optimizers import Adam


class Discriminator:
    """Class that defines the conditional discriminator of a Generative Adversarial Network (GAN).

    This class represents the conditional discriminator of a GAN, which is responsible for
    discriminating between real images and those generated by the generator, conditioned on some input.

    Attributes:
        input_shape (tuple): Shape of the input images to the discriminator.
        conditional_shape (tuple): Shape of the conditional input.
        model (keras.models.Model): Model of the discriminator.
    """

    def __init__(self, input_shape, text_embedding_dim):
        """Initializes the conditional discriminator with the specified input shapes.

        Args:
            input_shape (tuple): Shape of the input images to the discriminator.
            text_embedding_dim (int): Dimension of the GloVe word embeddings.
        """
        self.input_shape = input_shape
        self.text_embedding_dim = text_embedding_dim
        self.model = self.build_model()

    def build_model(self):
        """Builds and compiles the model of the conditional discriminator.

        Returns:
            keras.models.Model: Model of the conditional discriminator.
        """
        input_image = Input(shape=self.input_shape)
        conditional_input = Input(shape=(self.text_embedding_dim,))

        # Flatten the conditional input before passing it to RepeatVector
        conditional_input_flattened = Flatten()(conditional_input)
        conditional_input_repeated = RepeatVector(int(np.prod(self.input_shape[:-1])))(
            conditional_input_flattened
        )
        conditional_input_repeated = Reshape(
            self.input_shape[:-1] + (self.text_embedding_dim,)
        )(conditional_input_repeated)

        # Concatenate the input image and conditional input
        combined_input = Concatenate()([input_image, conditional_input_repeated])

        # Convolutional layers
        x = Conv2D(64, kernel_size=3, padding="same")(combined_input)
        x = LeakyReLU(negative_slope=0.2)(x)

        x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU(negative_slope=0.2)(x)

        x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU(negative_slope=0.2)(x)

        x = Conv2D(256, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU(negative_slope=0.2)(x)

        # Flattening layer
        x = Flatten()(x)

        # Dropout layer
        x = Dropout(0.4)(x)

        # Dense layer
        x = Dense(1, activation="sigmoid")(x)

        # Create the model
        model = Model(inputs=[input_image, conditional_input], outputs=x)

        # Optimizer
        optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

        # Compile the model
        model.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        return model

    def summary(self):
        """Prints a summary of the conditional discriminator model."""
        return self.model.summary()

    def evaluate(self, x, y):
        """Evaluates the conditional discriminator model on a set of input data and labels.

        Args:
            x (numpy.ndarray): Input data set.
            y (numpy.ndarray): True labels corresponding to the input data.

        Returns:
            list: List containing the loss and accuracy of the model on the
              input data set.
        """
        return self.model.evaluate(x, y)

    def trainable(self, trainable):
        """Sets whether the conditional discriminator model is trainable or not.

        Args:
            trainable (bool): Indicates whether the conditional discriminator model is
            trainable or not.
        """
        self.model.trainable = trainable
