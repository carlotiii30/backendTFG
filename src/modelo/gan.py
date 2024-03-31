from keras.models import Sequential
from keras.optimizers import Adam


class GAN:
    """
    The GAN (Generative Adversarial Network) class represents a GAN model.

    Attributes:
        discriminator (object): The discriminator model.
        generator (object): The generator model.
        gan (object): The compiled GAN model.

    Methods:
        __init__(discriminator, generator): Initializes the GAN class.
        create_gan(discriminator, generator): Creates the GAN model.
        summary(): Prints the summary of the GAN model.
    """

    def __init__(self, discriminator, generator):
        self.gan = self.create_gan(discriminator, generator)

    def create_gan(self, discriminator, generator):
        """
        Creates the GAN model.

        Args:
            discriminator (object): The discriminator model.
            generator (object): The generator model.

        Returns:
            object: The compiled GAN model.
        """
        discriminator.trainable = False
        gan = Sequential()
        gan.add(generator)
        gan.add(discriminator)

        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        gan.compile(loss="binary_crossentropy", optimizer=opt)

        return gan

    def summary(self):
        """
        Prints the summary of the GAN model.
        """
        self.gan.summary()
