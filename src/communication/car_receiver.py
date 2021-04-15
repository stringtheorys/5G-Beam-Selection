import argparse
import socket

from core.common import parse_model

parser = argparse.ArgumentParser()
parser.add_argument('-h', '--host', default='127.0.0.1')
parser.add_argument('-p', '--port', default=65432)


def start(args):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as receiver_socket:
        receiver_socket.connect((args.host, args.port))

        data = receiver_socket.recv(1024).decode('ascii').split(', ')
        training_type, model_type = data[0].replace('training: ', ''), data[1].replace('model: ', '')

        model = parse_model(model_type)
        receiver_socket.sendall(b'ready')

        while True:
            data = receiver_socket.recv(1024)
            model_variables = data.decode('magic')
            model.trainable_variables = model_variables

            if training_type == 'centralised':
                pass
            elif training_type == 'distributed':
                pass
            elif training_type == 'federated':
                pass


def start_car_receiver(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(b'Hello, world')
        data = s.recv(1024)

    print('Received', repr(data))


if __name__ == '__main__':
    parsed_arg = parser.parse_args()

    start(parsed_arg)
