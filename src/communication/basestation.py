
import socket


def start(model_name='imperial', num_vehicles=2):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as basestation_socket:
        basestation_socket.bind(('127.0.0,1', 12354))

        basestation_socket.listen(num_vehicles)
        vehicle_sockets = []

        print('Listening for connections')
        for _ in range(num_vehicles):
            vehicle_socket, addr = basestation_socket.accept()

            print('Successful connection')
            vehicle_socket.send(model_name.encode('utf8'))
            vehicle_sockets.append(vehicle_socket)
