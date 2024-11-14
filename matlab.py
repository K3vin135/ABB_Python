import socket

def start_server():
    host = 'localhost'
    port = 12345

    # Crear un socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print('Servidor Python esperando conexiones en el puerto', port)

    conn, addr = server_socket.accept()
    print('Conectado a:', addr)

    # Coordenadas iniciales y variables adicionales
    x, y, z = 233, -507, 500
    Moverobot = 1  # 1 para True, 0 para False
    numBox = 1

    try:
        # Crear mensaje con las coordenadas actuales y variables adicionales
        message = f"{x},{y},{z},{Moverobot},{numBox}\n"
        
        # Enviar coordenadas y variables adicionales al cliente MATLAB
        conn.send(message.encode('utf-8'))
        print('Enviado:', message.strip())

        # Cerrar la conexión después de enviar el mensaje una sola vez
        conn.close()
        server_socket.close()
    except Exception as e:
        print("Error:", e)
    finally:
        conn.close()
        server_socket.close()

if __name__ == "__main__":
    start_server()
