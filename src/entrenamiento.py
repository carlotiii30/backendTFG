from modelo.dataset import Dataset
from gan.discriminador import Discriminator

# Entrenamiento del discriminador

class Traininig:
    def train_discriminator(model, dataset, n_iters=20, batch=128):
        half_batch = int(batch / 2)

        for i in range(n_iters):
            X_real, y_real = dataset.load_real_data(dataset, half_batch)
            _, real_acc = model.train_on_batch(X_real, y_real)

            X_fake, y_fake = dataset.load_fake_data(half_batch)
            _, fake_acc = model.train_on_batch(X_fake, y_fake)

            print(f'Iteración: {i + 1}, Accuracy real: {real_acc * 100}, Accuracy fake: {fake_acc * 100}')

dataset = Dataset('cifar10')
#model = Discriminator((32, 32, 3))
#Traininig.train_discriminator(model, dataset)