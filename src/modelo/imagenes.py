import random
from PIL import Image
import io
import numpy as np


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