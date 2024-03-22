import unittest
from unittest.mock import patch
from src.conexion.servidor import Server


class TestConexion(unittest.TestCase):
    def setUp(self):
        self.server = Server("localhost", 12345)

    def test_server_creation(self):
        self.assertEqual(self.server.host, "localhost")
        self.assertEqual(self.server.port, 12345)


@patch("src.conexion.servidor.socket")
def test_server_start(self, mock_socket):
    mock_server_socket = mock_socket.socket.return_value
    mock_server_socket.accept.return_value = (
        "client_socket",
        "client_address",
    )

    self.server.start()

    mock_socket.socket.assert_called_once_with(
        mock_socket.AF_INET, mock_socket.SOCK_STREAM
    )
    mock_server_socket.bind.assert_called_once_with(("localhost", 12345))
    mock_server_socket.listen.assert_called_once()
    mock_server_socket.accept.assert_called_once()
    mock_socket.socket.return_value.close.assert_called_once()

    @patch("src.conexion.servidor.Handler")
    def test_client_handler(self, mock_handler):
        mock_client_socket = "client_socket"

        self.server.client_handler(mock_client_socket)

        mock_handler.assert_called_once_with(mock_client_socket)
        mock_handler.return_value.handle.assert_called_once()


if __name__ == "__main__":
    unittest.main()
