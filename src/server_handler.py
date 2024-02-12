import json
from imagenes import imagenes
import base64

"""
Clase que maneja las peticiones del cliente
"""


class server_handler:
    """
    Constructor de la clase
    :param client_socket: socket - Socket del cliente
    """
    def __init__(self, client_socket):
        self.client_socket = client_socket

    """
    Método que maneja la petición del cliente
    """
    def handle(self):
        with self.client_socket:
            data = self.client_socket.recv(1024).decode()
            try:
                request = json.loads(data)
                command = request.get("command")
                params = request.get("params", {})
                text = request.get("text", "")

                if command == "generate_image":
                    if text == "":
                        # Genera la imagen y conviértela en una cadena base64
                        image_bytes = imagenes.generar_imagen_random()
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

                    else:
                        # Genera la imagen y conviértela en una cadena base64
                        image_bytes = imagenes.generar_imagen(text)
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

                    response = {
                        "status": "success",
                        "message": "Imagen generada correctamente",
                        "image": image_base64,
                    }

                elif command == "suma":
                    response = {
                        "status": "success",
                        "result": params.get("a", 0) + params.get("b", 0),
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
                self.client_socket.sendall(json.dumps(response).encode())
            except Exception as e:
                print(f"Error al enviar la respuesta al cliente: {str(e)}")