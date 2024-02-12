import json
from imagenes import Imagenes
import base64

"""
Clase que maneja las peticiones del cliente
"""


class Manejador:
    """
    Constructor de la clase
    :param socket: socket - Socket del cliente
    """
    def __init__(self, socket):
        self.socket = socket

    """
    Método que maneja la petición del cliente
    """
    def manejar(self):
        with self.socket:
            data = self.socket.recv(1024).decode()
            try:
                request = json.loads(data)
                command = request.get("command")
                text = request.get("text", "")

                if command == "generar_imagen":
                    if text == "":
                        # Genera la imagen y conviértela en una cadena base64
                        image_bytes = Imagenes.generar_imagen_random()
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

                    else:
                        # Genera la imagen y conviértela en una cadena base64
                        image_bytes = Imagenes.generar_imagen(text)
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

                    response = {
                        "status": "success",
                        "message": "Imagen generada correctamente",
                        "image": image_base64,
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