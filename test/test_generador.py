import json
import unittest
from unittest.mock import Mock, patch
from pytest_mock import mocker
from src.conexion.manejador import Handler
import numpy as np


class TestGenerator(unittest.TestCase):
    def test_handle_generar_imagen_success(self):
        # Creamos un socket falso para simular la comunicación con el cliente
        mock_socket = Mock()

        # Creamos un objeto Handler con el socket falso
        handler = Handler(mock_socket)

        # Simulamos recibir una petición JSON válida para generar una imagen
        request = {"command": "generar_imagen"}
        mock_socket.recv.return_value.decode.return_value = json.dumps(request)

        # Simulamos que la generación de la imagen es exitosa
        mock_model = Mock()
        mock_model.return_value = np.ones((32, 32, 3))
        with patch("gan.generador.Generador", return_value=mock_model):
            handler.handle()

        # Verificamos que se envíe una respuesta con estado "success" y un mensaje adecuado
        mock_socket.sendall.assert_called_with(
            json.dumps(
                {
                    "status": "success",
                    "message": "Imagen generada correctamente",
                    "image": mocker.ANY,
                }
            ).encode()
        )

    def test_handle_procesar_texto_success(self):
        # Creamos un socket falso para simular la comunicación con el cliente
        mock_socket = Mock()

        # Creamos un objeto Handler con el socket falso
        handler = Handler(mock_socket)

        # Simulamos recibir una petición JSON válida para procesar un texto
        request = {"command": "procesar_texto", "text": "Hola mundo"}
        mock_socket.recv.return_value.decode.return_value = json.dumps(request)

        # Simulamos que el procesamiento del texto es exitoso
        with patch("modelo.procesamiento.Texto.procesar_texto") as mock_procesar_texto:
            mock_procesar_texto.return_value = (
                "Texto procesado",
                ["Hola", "mundo"],
                [1, 2, 3],
            )
            handler.handle()

        # Verificamos que se envíe una respuesta con estado "success" y un mensaje adecuado
        mock_socket.sendall.assert_called_with(
            json.dumps(
                {
                    "status": "success",
                    "message": "Texto procesado correctamente",
                    "text": "Texto procesado",
                    "tokens": ["Hola", "mundo"],
                    "representacion_numerica": [1, 2, 3],
                }
            ).encode()
        )


if __name__ == "__main__":
    unittest.main()
