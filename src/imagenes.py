import random
from PIL import Image
import io

class imagenes:

    def generar_imagen_random():
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
        image.save(img_byte_array, format='PNG')
        img_byte_array = img_byte_array.getvalue()

        return img_byte_array

    def generar_imagen(text):
        # Generar una imagen a partir de un texto
        image = Image.new("RGB", (500, 500), (255, 255, 255))
        pixels = image.load()

        for i in range(image.width):
            for j in range(image.height):
                r = ord(text[i % len(text)]) % 255
                g = ord(text[j % len(text)]) % 255
                b = (ord(text[i % len(text)]) + ord(text[j % len(text)])) % 255

                pixels[i, j] = (r, g, b)

        # Convertir la imagen a bytes
        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format='PNG')
        img_byte_array = img_byte_array.getvalue()

        return img_byte_array