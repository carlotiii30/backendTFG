import json

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

                if command == "generate_image":
                    # Aquí iría la lógica para generar la imagen
                    response = {
                        "status": "success",
                        "message": "Imagen generada correctamente",
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

            self.client_socket.sendall(json.dumps(response).encode())
