from textblob import TextBlob
import re


class Texto:
    """Clase que proporciona funciones para procesar texto.

    Esta clase contiene métodos para realizar operaciones de preprocesamiento y representación de texto.

    """

    @staticmethod
    def procesar_texto(texto):
        """Procesa un texto dado.

        Realiza operaciones de preprocesamiento, tokenización y representación del texto.

        Args:
            texto (str): El texto a procesar.

        Returns:
            str: El texto preprocesado.
            list: Lista de tokens obtenidos del texto.
            dict: Representación numérica del texto utilizando one-hot encoding.
        """
        # Preprocesamiento del texto
        texto = re.sub(r"[^\w\s]", "", texto)  # Eliminar caracteres especiales
        texto = texto.lower()  # Convertir a minúsculas

        # Tokenización
        tokens = TextBlob(texto).words

        # Representación del texto (one-hot encoding)
        vocabulario = set(tokens)
        representacion_numerica = {}
        for i, token in enumerate(vocabulario):
            representacion_numerica[token] = [1 if token == t else 0 for t in tokens]

        return texto, tokens, representacion_numerica
