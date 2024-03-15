from keras.datasets import cifar100
import numpy as np
from src.gan.discriminador import Discriminator

class Training:
    def load_images(dataset):
        (Xtrain, Ytrain), (_, _) = dataset.load_data()

        indice = np.where(Ytrain == 0)
        indice = indice[0]
        Xtrain = Xtrain[indice, :, :, :]

        X = Xtrain.astype('float32')
        X = (X - 127.5) / 127.5

        return X

    def load_real_data(dataset, n_samples):
        ix = np.random.randint(0, dataset.shape[0], n_samples)
        X = dataset[ix]
        y = np.ones((n_samples, 1))
        return X, y

    def load_fake_data(n_samples):
        X = np.random.rand(32 * 32 * 3 * n_samples)
        X = -1 + X * 2
        X = X.reshape((n_samples, 32, 32, 3))
        y = np.zeros((n_samples, 1))
        return X, y

    def train_discriminator(discriminator, dataset, n_iter=20, n_batch=128):
        half_batch = int(n_batch / 2)

        for i in range(n_iter):
            X_real, y_real = Training.load_real_data(dataset, half_batch)
            _, real_acc = discriminator.model.train_on_batch(X_real, y_real)

            X_fake, y_fake = Training.load_fake_data(half_batch)
            _, fake_acc = discriminator.model.train_on_batch(X_fake, y_fake)

            print(f"{i + 1}, Real={real_acc*100}, Fake={fake_acc*100}")