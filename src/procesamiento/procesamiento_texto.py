from textblob import TextBlob
import re
import numpy as np


class Texto:
    """Clase que proporciona funciones para procesar texto.

    Esta clase contiene métodos para realizar operaciones de preprocesamiento y
    representación de texto.

    """

    @staticmethod
    def procesar_texto(texto):
        """Procesa un texto dado.

        Realiza operaciones de preprocesamiento, tokenización y representación
        del texto.

        Args:
            texto (str): El texto a procesar.

        Returns:
            str: El texto preprocesado.
            list: Lista de tokens obtenidos del texto.
            np.array: Representación numérica del texto utilizando one-hot encoding.
        """
        # Preprocesamiento del texto
        texto_preprocesado = re.sub(
            r"[^\w\s]", "", texto
        )  # Eliminar caracteres especiales
        texto_preprocesado = texto_preprocesado.lower()  # Convertir a minúsculas

        # Tokenización
        tokens = TextBlob(texto_preprocesado).words

        # Construir vocabulario
        vocabulario = []
        for token in tokens:
            if token not in vocabulario:
                vocabulario.append(token)

        # Representación del texto (one-hot encoding)
        representacion_numerica = np.zeros((len(tokens), len(vocabulario)), dtype=int)
        for i, token in enumerate(tokens):
            representacion_numerica[i, vocabulario.index(token)] = 1

        return texto_preprocesado, tokens, representacion_numerica
