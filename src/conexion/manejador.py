import json
import numpy as np
import logging
import base64
from PIL import Image
from src.procesamiento.procesamiento_texto import Text
from src.modelo.componentes.generador import Generator


class Handler:
    """Class that handles client requests.

    This class handles the requests sent by the client and executes the
    corresponding actions.

    Attributes:
        socket (socket): Client socket.
    """

    def __init__(self, socket):
        """Class constructor.

        Args:
            socket: Client socket.
        """
        self.socket = socket

    def handle(self):
        """Method that handles the client's request."""
        with self.socket:
            data = self.socket.recv(1024).decode()
            try:
                request = json.loads(data)
                command = request.get("command")
                text = request.get("text", "")

                if command == "generate_image":
                    try:
                        _, _, numeric_representation = Text.process_text(text)
                        numeric_representation = np.array(numeric_representation)
                        numeric_representation = np.mean(
                            numeric_representation, axis=0
                        ).reshape(1, -1)

                        dim = numeric_representation.shape[1]

                        generated_images = Generator(100, dim, (32, 32, 3)).predict(
                            [
                                np.random.randn(1, 100),
                                numeric_representation,
                            ]
                        )

                        scaled_images = ((generated_images + 1) * 127.5).astype(np.uint8)
                        image64 = base64.b64encode(
                            Image.fromarray(scaled_images[0]).tobytes()
                        ).decode()

                        response = {
                            "status": "success",
                            "message": "Image generated successfully",
                            "image": image64,
                        }

                        logging.info("Image generated successfully")

                    except Exception as e:
                        response = {
                            "status": "error",
                            "message": f"Error generating the image: {str(e)}",
                        }

                        logging.error(f"Error generating the image: {str(e)}")

                elif command == "process_text":
                    try:
                        # Process the text using the Text class
                        text, tokens, numeric_representation = Text.process_text(
                            text
                        )

                        response = {
                            "status": "success",
                            "message": "Text processed successfully",
                            "text": text,
                            "tokens": tokens,
                            "numeric_representation": numeric_representation,
                        }

                        logging.info("Text processed successfully")

                    except Exception as e:
                        response = {
                            "status": "error",
                            "message": f"Error processing the text: {str(e)}",
                        }

                        logging.error(f"Error processing the text: {str(e)}")

                else:
                    response = {
                        "status": "error",
                        "message": f"Unknown command: {command}",
                    }

                    logging.error(f"Unknown command: {command}")

            except json.JSONDecodeError as e:
                response = {
                    "status": "error",
                    "message": f"Error decoding JSON: {str(e)}",
                }

                logging.error(f"Error decoding JSON: {str(e)}")

            except Exception as e:
                response = {
                    "status": "error",
                    "message": f"Error: {str(e)}",
                }

                logging.error(f"Error: {str(e)}")

            try:
                self.socket.sendall(json.dumps(response).encode())

                logging.info("Response sent to the client")

            except Exception as e:
                print(f"Error sending the response to the client: {str(e)}")

                logging.error(f"Error sending the response to the client: {str(e)}")
