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
    def load_fake_data(n_samples):
        """
        Generate fake data samples.

        Args:
            n_samples: The number of fake data samples to generate.

        Returns:
            The fake data samples and their corresponding labels.
        """
        X = np.random.rand(n_samples, 32, 32, 3)
        X = -1 + X * 2
        y = np.zeros((n_samples, 1))
        return X, y

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
    def train_discriminator(discriminator, dataset, condition, n_iter=20, n_batch=128):
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
            X_real, y_real = Training.load_real_data(dataset, half_batch)
            # Agrega la condición a los datos reales
            X_real = np.concatenate([X_real, condition], axis=1)
            real_acc = Training.train_step(discriminator.model, X_real, y_real)

            X_fake, y_fake = Training.load_fake_data(half_batch)
            # Agrega la condición a los datos falsos
            X_fake = np.concatenate([X_fake, condition], axis=1)
            fake_acc = Training.train_step(discriminator.model, X_fake, y_fake)

            print(
                f"Epoch: {i + 1}, Real Accuracy: {real_acc * 100}, Fake Accuracy: {fake_acc * 100}, Condition: {condition}"
            )
