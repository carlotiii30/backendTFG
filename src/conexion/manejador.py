import json
import numpy as np
import logging
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
                        _, _, n1, n2 = Text.process_text(text)

                        n = np.concatenate((n1, n2), axis=1)
                        n = n[:, :9]

                        dim = n.shape[1]

                        gen = Generator(100, dim, (32, 32, 3))
                        generated_images = gen.predict([np.random.randn(1, 100), n])

                        generated_images = (
                            generated_images - generated_images.min()
                        ) / (generated_images.max() - generated_images.min())

                        image_list = generated_images[0].tolist()

                        image_json = json.dumps(image_list)

                        response = {
                            "status": "success",
                            "message": "Image generated successfully",
                            "image": image_json,
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
                        text, tokens, numeric_representation = Text.process_text(text)

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
