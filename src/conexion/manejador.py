import json
import numpy as np
from src.prueba.imagenes import Images
import base64
from src.procesamiento.procesamiento_texto import Texto
from src.gan.generador import Generador

"""
Clase que maneja las peticiones del cliente
"""


class Handler:
    """
    Constructor de la clase
    :param socket: socket - Socket del cliente
    """

    def __init__(self, socket):
        self.socket = socket

    """
    Método que maneja la petición del cliente
    """

    def handle(self):
        with self.socket:
            data = self.socket.recv(1024).decode()
            try:
                request = json.loads(data)
                command = request.get("command")
                text = request.get("text", "")

                if command == "generar_imagen":
                    """imagen = Images.generate_random()
                    imagen64 = base64.b64encode(imagen).decode()"""

                    try:
                        gen_model = Generador(100, (32, 32, 3))
                        image = Images.generate_image(gen_model.model, 1)
                        image = (image * 255).astype(np.uint8)
                        imagen64 = base64.b64encode(image).decode()

                        response = {
                            "status": "success",
                            "message": "Imagen generada correctamente",
                            "image": imagen64,
                        }

                    except Exception as e:
                        response = {
                            "status": "error",
                            "message": f"Error al generar la imagen: {str(e)}",
                        }

                elif command == "procesar_texto":

                    try:
                        # Procesa el texto
                        text, tokens, representacion_numerica = Texto.procesar_texto(text)

                        response = {
                            "status": "success",
                            "message": "Texto procesado correctamente",
                            "text": text,
                            "tokens": tokens,
                            "representacion_numerica": representacion_numerica,
                        }
                    except Exception as e:
                        response = {
                            "status": "error",
                            "message": f"Error al procesar el texto: {str(e)}",
                        }

                else:
                    response = {
                        "status": "error",
                        "message": f"Comando desconocido: {command}",
                    }

            except json.JSONDecodeError as e:
                response = {
                    "status": "error",
                    "message": "Error al decodificar el JSON",
                }

            except Exception as e:
                response = {
                    "status": "error",
                    "message": f"Error: {str(e)}",
                }

            try:
                self.socket.sendall(json.dumps(response).encode())
            except Exception as e:
                print(f"Error al enviar la respuesta al cliente: {str(e)}")