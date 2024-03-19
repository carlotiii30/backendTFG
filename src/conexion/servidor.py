import socket
import threading
from src.conexion.manejador import Handler


class Server:
    """Clase que representa un servidor que escucha en un puerto y maneja las
    conexiones de los clientes.

    Esta clase se encarga de iniciar un servidor que escucha en un puerto
    específico y maneja las conexiones entrantes de los clientes.

    Attributes:
        host (str): Dirección IP o nombre del host donde escuchará el servidor.
        port (int): Puerto donde escuchará el servidor.
    """

    def __init__(self, host, port):
        """Constructor de la clase.

        Args:
            host (str): Dirección IP o nombre del host donde escuchará el
            servidor.
            port (int): Puerto donde escuchará el servidor.
        """
        self.host = host
        self.port = port

    def start(self):
        """Método que inicia el servidor."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((self.host, self.port))
            server_socket.listen()
            print(f"Servidor escuchando en el puerto {self.port}")
            while True:
                client_socket, _ = server_socket.accept()
                threading.Thread(
                    target=self.client_handler, args=(client_socket,)
                ).start()

    def client_handler(self, client_socket):
        """Método que maneja la conexión con un cliente.

        Args:
            client_socket (socket): Socket del cliente.
        """
        handler = Handler(client_socket)
        handler.handle()
