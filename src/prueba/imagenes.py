import random
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt


class Images:

    def generate_random():
        # Generar una imagen aleatoria
        image = Image.new("RGB", (500, 500))
        pixels = image.load()

        for i in range(image.width):
            for j in range(image.height):
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)

                pixels[i, j] = (r, g, b)

        # Convertir la imagen a bytes
        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format="PNG")
        img_byte_array = img_byte_array.getvalue()

        return img_byte_array

    def generate(texto):
        # Generar una imagen a partir de un texto
        image = Image.new("RGB", (500, 500), (255, 255, 255))
        pixels = image.load()

        for i in range(image.width):
            for j in range(image.height):
                r = ord(texto[i % len(texto)]) % 255
                g = ord(texto[j % len(texto)]) % 255
                b = (ord(texto[i % len(texto)]) + ord(texto[j % len(texto)])) % 255

                pixels[i, j] = (r, g, b)

        # Convertir la imagen a bytes
        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format="PNG")
        img_byte_array = img_byte_array.getvalue()

        return img_byte_array

    def generate_input_data(n_samples):
        X = np.random.randn(100 * n_samples)
        X = X.reshape((n_samples, 100))
        return X

    def create_fake_data(model, n_samples):
        input = Images.generate_input_data(n_samples)
        X = model.predict(input)
        y = np.zeros((n_samples, 1))
        return X, y

    def generate_image(model, n_samples):
        X, _ = Images.create_fake_data(model, n_samples)

        for i in range(n_samples):
            # Escalar valores de píxeles (0, 255)
            image_data = np.clip(X[i] * 255, 0, 255).astype(np.uint8)
            # Crear imagen
            image = Image.fromarray(image_data)

        return image
