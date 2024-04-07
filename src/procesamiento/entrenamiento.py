import numpy as np


class Training:
    """
    A class that contains methods for training a model.
    """

    @staticmethod
    def load_images(dataset):
        """
        Load and preprocess images from a dataset.

        Args:
            dataset: The dataset to load the images from.

        Returns:
            The preprocessed images.
        """
        (Xtrain, _), (_, _) = dataset.load_data()

        X = Xtrain.astype("float32")
        X = (X - 127.5) / 127.5

        return X

    @staticmethod
    def load_real_data(dataset, n_samples):
        """
        Load real data samples from a dataset.

        Args:
            dataset: The dataset to load the real data from.
            n_samples: The number of samples to load.

        Returns:
            The real data samples and their corresponding labels.
        """
        ix = np.random.randint(0, dataset.shape[0], n_samples)
        X = dataset[ix]
        y = np.ones((n_samples, 1))
        return X, y

    @staticmethod
    def generate_fake_data(generator, condition, n_samples):
        """
        Generate fake data with a conditional generator.

        Args:
            generator: The conditional generator model.
            condition: The condition to use when generating the fake data.
            n_samples: The number of samples to generate.

        Returns:
            The generated fake samples and their corresponding labels.
        """
        # Genera un vector de ruido aleatorio
        z_noise = np.random.normal(0, 1, (n_samples, generator.latent_dim))

        # Genera las imágenes falsas a partir del ruido y la condición
        images = generator.predict([z_noise, np.repeat(condition, n_samples, axis=0)])

        # Crea las etiquetas para las imágenes falsas
        y = np.zeros((n_samples, 1))

        return images, y

    @staticmethod
    def train_step(model, X, y):
        """
        Perform a single training step on the model.

        Args:
            model: The model to train.
            X: The input data.
            y: The target labels.

        Returns:
            The accuracy of the training step.
        """
        _, acc = model.train_on_batch(X, y)
        return acc

    @staticmethod
    def train_discriminator(
        self, discriminator, dataset, condition, n_iter=20, n_batch=128
    ):
        """
        Train the discriminator model.

        Args:
            discriminator: The discriminator model to train.
            dataset: The dataset to train the discriminator on.
            condition: The additional condition for the discriminator.
            n_iter: The number of training iterations.
            n_batch: The batch size.

        Returns:
            None
        """
        half_batch = int(n_batch / 2)

        for i in range(n_iter):
            # Carga y prepara los datos reales
            X_real, y_real = self.load_real_data(dataset, half_batch)
            # Agrega la condición a los datos reales
            X_real = [X_real, np.repeat(condition, half_batch, axis=0)]
            real_acc = self.train_step(discriminator.model, X_real, y_real)

            # Genera y prepara los datos falsos
            X_fake, y_fake = self.generate_fake_data(half_batch)
            # Agrega la condición a los datos falsos
            X_fake = [X_fake, np.repeat(condition, half_batch, axis=0)]
            fake_acc = self.train_step(discriminator.model, X_fake, y_fake)

            print(
                f"Epoch: {i + 1}, Real Accuracy: {real_acc * 100}, Fake Accuracy: {fake_acc * 100}, Condition: {condition}"
            )
