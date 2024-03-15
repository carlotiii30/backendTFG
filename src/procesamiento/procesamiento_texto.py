# Procesamiento de texto
from textblob import TextBlob
import re

class Texto:
    def procesar_texto(texto):
        # Preprocesamiento del texto
        texto = re.sub(r'[^\w\s]', '', texto)   # Eliminar caracteres especiales
        texto = texto.lower()                   # Convertir a minúsculas

        tokens = TextBlob(texto).words          # Tokenización

        # Representación del texto (one-hot encoding)
        vocabulario = set(tokens)
        representacion_numerica = {}
        for i, token in enumerate(vocabulario):
            representacion_numerica[token] = [1 if token == t else 0 for t in tokens]

        return texto, tokens, representacion_numerica