# Procesamiento de texto
from textblob import TextBlob
import re

# Texto de ejemplo
texto = "Este es un ejemplo de texto que contiene caracteres especiales, mayúsculas y palabras vacías como 'el', 'de' y 'un'."

# Preprocesamiento del texto
texto = re.sub(r'[^\w\s]', '', texto)  # Eliminar caracteres especiales
texto = texto.lower()  # Convertir a minúsculas

# Tokenización
tokens = TextBlob(texto).words

# Representación del texto (one-hot encoding)
vocabulario = set(tokens)  # Construir el vocabulario
representacion_numerica = {}
for i, token in enumerate(vocabulario):
    representacion_numerica[token] = [1 if token == t else 0 for t in tokens]

print("Texto preprocesado:", texto, "\n")
print("Tokens:", tokens, "\n")
print("Representación numérica:", representacion_numerica, "\n")