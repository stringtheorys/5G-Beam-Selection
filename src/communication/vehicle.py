
import socket

from models import models


def start():
    vehicle_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Trying to connect to the basestation')
    vehicle_socket.connect(('localhost', 12354))

    print('Successful connect; now waiting for model name')
    model_name = vehicle_socket.recv(255).decode('utf8')
    print(f'Model name is {model_name}')

    mode_fn, dataset_fn = models[model_name]

    # Todo receive new global model, update model and send back the updated model
