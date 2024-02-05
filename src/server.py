import socket

# Configurar el servidor
HOST = 'localhost'
PORT = 12345

# Crear un socket TCP/IP
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    try:
        # Vincular el socket a la dirección y el puerto especificados
        server_socket.bind((HOST, PORT))
        # Escuchar conexiones entrantes
        server_socket.listen()

        print("Esperando conexiones entrantes en el puerto", PORT)

        # Aceptar la conexión entrante
        client_socket, client_address = server_socket.accept()

        with client_socket:
            print('Conexión establecida desde', client_address)
            while True:
                # Recibir datos del cliente
                data = client_socket.recv(1024)
                if not data:
                    break
                print('Mensaje recibido:', data.decode())
    except Exception as e:
        print("Error en el servidor:", e)
