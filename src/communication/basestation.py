
import socket
import argparse
from pickle import dumps

from core.common import parse_model

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--training', default='distributed', choices=['centralised', 'distributed', 'federated'])
parser.add_argument('-m', '--model', default='imperial')  # Add choices
parser.add_argument('-e', '--epochs', defaults=1000)
parser.add_argument('-h', '--host', default='127.0.0.1')
parser.add_argument('-p', '--port', default=65432)


def start(args):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as basestation_socket:
        basestation_socket.bind((args.host, args.port))

        print('[+] Base station is listening')
        basestation_socket.listen()

        conn, addr = basestation_socket.accept()
        print(f'[+] Connection accepted - {addr}')

        model, *train_validation_data = parse_model(args.model)
        conn.sendall(f'training: {args.training}, model: {args.model}'.encode('ascii'))

        conn.recv(1024)
        for epoch in range(args.epochs):
            conn.sendall(dumps(model.trainable_variables))

            data = conn.recv(1024)
            if not data:
                break

            if args.training == 'centralised':
                pass
            elif args.training == 'distributed':
                pass
            elif args.training == 'federated':
                pass
        conn.sendall(b'stop')


if __name__ == '__main__':
    parsed_args = parser.parse_args()

    start(parsed_args)
