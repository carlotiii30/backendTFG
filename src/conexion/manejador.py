import json
import numpy as np
import logging
import base64
from PIL import Image
from src.procesamiento.procesamiento_texto import Text
from src.modelo.componentes.generador import Generator


class Handler:
    """Clase que maneja las peticiones del cliente.

    Esta clase maneja las peticiones enviadas por el cliente y ejecuta las
    acciones correspondientes.

    Attributes:
        socket (socket): Socket del cliente.
    """

    def __init__(self, socket):
        """Constructor de la clase.

        Args:
            socket: Socket del cliente.
        """
        self.socket = socket

    def handle(self):
        """Método que maneja la petición del cliente."""
        with self.socket:
            data = self.socket.recv(1024).decode()
            try:
                request = json.loads(data)
                command = request.get("command")
                text = request.get("text", "")

                if command == "generar_imagen":
                    try:
                        _, _, representacion_numerica = Text.process_text(text)
                        representacion_numerica = np.array(representacion_numerica)
                        representacion_numerica = np.mean(
                            representacion_numerica, axis=0
                        ).reshape(1, -1)

                        dim = representacion_numerica.shape[1]

                        generated_images = Generator(100, dim, (32, 32, 3)).predict(
                            [
                                np.random.randn(1, 100),
                                representacion_numerica,
                            ]
                        )

                        scaled_images = ((generated_images + 1) * 127.5).astype(np.uint8)
                        image64 = base64.b64encode(
                            Image.fromarray(scaled_images[0]).tobytes()
                        ).decode()

                        response = {
                            "status": "success",
                            "message": "Imagen generada correctamente",
                            "image": image64,
                        }

                        logging.info("Imagen generada correctamente")

                    except Exception as e:
                        response = {
                            "status": "error",
                            "message": f"Error al generar la imagen: {str(e)}",
                        }

                        logging.error(f"Error al generar la imagen: {str(e)}")

                elif command == "procesar_texto":
                    try:
                        # Procesa el texto utilizando la clase Texto
                        text, tokens, representacion_numerica = Text.process_text(
                            text
                        )

                        response = {
                            "status": "success",
                            "message": "Texto procesado correctamente",
                            "text": text,
                            "tokens": tokens,
                            "representacion_numerica": representacion_numerica,
                        }

                        logging.info("Texto procesado correctamente")

                    except Exception as e:
                        response = {
                            "status": "error",
                            "message": f"Error al procesar el texto: {str(e)}",
                        }

                        logging.error(f"Error al procesar el texto: {str(e)}")

                else:
                    response = {
                        "status": "error",
                        "message": f"Comando desconocido: {command}",
                    }

                    logging.error(f"Comando desconocido: {command}")

            except json.JSONDecodeError as e:
                response = {
                    "status": "error",
                    "message": f"Error al decodificar el JSON: {str(e)}",
                }

                logging.error(f"Error al decodificar el JSON: {str(e)}")

            except Exception as e:
                response = {
                    "status": "error",
                    "message": f"Error: {str(e)}",
                }

                logging.error(f"Error: {str(e)}")

            try:
                self.socket.sendall(json.dumps(response).encode())

                logging.info("Respuesta enviada al cliente")

            except Exception as e:
                print(f"Error al enviar la respuesta al cliente: {str(e)}")

                logging.error(f"Error al enviar la respuesta al cliente: {str(e)}")
