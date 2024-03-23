import numpy as np


class Training:
    @staticmethod
    def load_images(dataset):
        (Xtrain, _), (_, _) = dataset.load_data()

        X = Xtrain.astype("float32")
        X = (X - 127.5) / 127.5

        return X

    @staticmethod
    def load_real_data(dataset, n_samples):
        ix = np.random.randint(0, dataset.shape[0], n_samples)
        X = dataset[ix]
        y = np.ones((n_samples, 1))
        return X, y

    @staticmethod
    def load_fake_data(n_samples):
        X = np.random.rand(n_samples, 32, 32, 3)
        X = -1 + X * 2
        y = np.zeros((n_samples, 1))
        return X, y

    @staticmethod
    def train_step(model, X, y):
        _, acc = model.train_on_batch(X, y)
        return acc

    @staticmethod
    def train_discriminator(discriminator, dataset, n_iter=20, n_batch=128):
        half_batch = int(n_batch / 2)

        for i in range(n_iter):
            X_real, y_real = Training.load_real_data(dataset, half_batch)
            real_acc = Training.train_step(discriminator.model, X_real, y_real)

            X_fake, y_fake = Training.load_fake_data(half_batch)
            fake_acc = Training.train_step(discriminator.model, X_fake, y_fake)

            print(
                f"Epoch: {i + 1}, Real Accuracy: {real_acc * 100}, Fake Accuracy: {fake_acc * 100}"
            )
