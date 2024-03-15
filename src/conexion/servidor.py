import socket
import threading
from src.conexion.manejador import Handler

"""
Clase que representa un servidor que escucha en un puerto y maneja las conexiones de los clientes
"""
class Server:
    """
    Constructor de la clase
    :param host: str - Dirección IP o nombre del host donde escuchará el servidor
    :param port: int - Puerto donde escuchará el servidor
    """
    def __init__(self, host, port):
        self.host = host
        self.port = port

    """
    Método que inicia el servidor
    """
    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((self.host, self.port))
            server_socket.listen()
            print(f"Servidor escuchando en el puerto {self.port}")
            while True:
                client_socket, _ = server_socket.accept()
                threading.Thread(
                    target=self.client_handler, args=(client_socket,)
                ).start()

    """
    Método que maneja la conexión con un cliente
    :param client_socket: socket - Socket del cliente
    """
    def client_handler(self, client_socket):
        handler = Handler(client_socket)
        handler.handle()
