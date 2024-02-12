import socket
import threading
from manejador import Manejador

"""
Clase que representa un servidor que escucha en un puerto y maneja las conexiones de los clientes
"""
class Servidor:
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
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_servidor:
            socket_servidor.bind((self.host, self.port))
            socket_servidor.listen()
            print(f"Servidor escuchando en el puerto {self.port}")
            while True:
                socket_cliente, _ = socket_servidor.accept()
                threading.Thread(
                    target=self.manejador_cliente, args=(socket_cliente,)
                ).start()

    """
    Método que maneja la conexión con un cliente
    :param socket_cliente: socket - Socket del cliente
    """
    def manejador_cliente(self, socket_cliente):
        manejador = Manejador(socket_cliente)
        manejador.manejar()
