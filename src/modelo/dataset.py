from keras.datasets import cifar10, cifar100
import random
import numpy as np

class Dataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        if dataset_name == 'cifar10':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        elif dataset_name == 'cifar100':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar100.load_data()
        else:
            raise ValueError('Invalid dataset name')

    def get_train_data(self):
        return self.x_train, self.y_train

    def get_test_data(self):
        return self.x_test, self.y_test

    def load_real_data(dataset, n_samples):
        ix = random.randint(0, dataset.shape[0], n_samples)
        X = dataset[ix]
        y = np.ones((n_samples, 1))

        return X, y

    def load_fake_data(n_samples):
        X = np.random.rand(32 * 32 * 3 * n_samples)
        X = -1 + X * 2
        X = X.reshape((n_samples, 32, 32, 3))
        y = np.zeros((n_samples, 1))

        return X, y