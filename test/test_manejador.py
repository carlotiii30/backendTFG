from src.conexion.manejador import Handler
import json
import pytest

class TestHandler:

    # Handler receives valid JSON request with "generar_imagen" command and generates a random image and returns it as a base64 encoded string
    def test_generar_imagen_random(self, mocker):
        # Mock the socket object
        mock_socket = mocker.Mock()
        # Create a Handler instance
        handler = Handler(mock_socket)
        # Create a valid JSON request with "generar_imagen" command
        request = {
            "command": "generar_imagen"
        }
        # Convert the request to JSON string
        request_json = json.dumps(request)
        # Set the return value of socket.recv() to the request JSON string
        mock_socket.recv.return_value.decode.return_value = request_json

        # Invoke the handle() method of the Handler instance
        handler.handle()

        # Assert that socket.sendall() was called with the correct response JSON
        mock_socket.sendall.assert_called_with(json.dumps({
            "status": "success",
            "message": "Imagen generada correctamente",
            "image": mocker.ANY,
        }).encode())

    # Handler receives valid JSON request with "generar_imagen" command and generates an image using a fake model with n_samples=0
    def test_generar_imagen_fake_model_n_samples_0(self, mocker):
        # Mock the socket object
        mock_socket = mocker.Mock()
        # Create a Handler instance
        handler = Handler(mock_socket)
        # Create a valid JSON request with "generar_imagen" command and n_samples=0
        request = {
            "command": "generar_imagen",
            "n_samples": 0
        }
        # Convert the request to JSON string
        request_json = json.dumps(request)
        # Set the return value of socket.recv() to the request JSON string
        mock_socket.recv.return_value.decode.return_value = request_json

        # Invoke the handle() method of the Handler instance
        handler.handle()

        # Assert that socket.sendall() was called with the correct response JSON
        mock_socket.sendall.assert_called_with(json.dumps({
            "status": "success",
            "message": "Imagen generada correctamente",
            "image": mocker.ANY,
        }).encode())