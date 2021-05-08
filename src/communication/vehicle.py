"""
Implementation of a 5G vehicle
"""

import json
import os
import pickle
import socket

import tensorflow as tf

from core.dataset import output_dataset
from core.metrics import TopKThroughputRatio
from models import models


def start(num_vehicles=2):
    vehicle_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Trying to connect to the basestation')
    vehicle_socket.connect(('127.0.0.1', 12354))

    print('Successful connect; now waiting for model name')
    model_name = vehicle_socket.recv(255).decode('utf8')
    print(f'Model name is {model_name}')

    metrics = [
        tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top-1-accuracy'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top-10-accuracy'),
        TopKThroughputRatio(k=1, name='top-1-throughput'),
        TopKThroughputRatio(k=10, name='top-10-throughput')
    ]

    model_fn, dataset_fn = models[model_name]
    local_model = model_fn()
    local_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(),
                        metrics=metrics)
    model_recv_size = len(pickle.dumps(local_model.trainable_variables))

    training_input, _ = dataset_fn()
    training_output, _ = output_dataset()

    indexes = tf.random.uniform((512,), 0, len(training_input))
    training_input = tf.gather(training_input, indexes) if not isinstance(training_input, tuple) else \
        tuple(tf.gather(dataset_input, indexes) for dataset_input in training_input)
    training_output = tf.gather(training_output, indexes)

    # receive new global model, update model and send back the updated model
    vehicle_results = []
    while True:
        received_data = vehicle_socket.recv(model_recv_size)
        if not received_data:
            break

        updated_trainable_variables = pickle.loads(received_data)
        for local_var, updated_var in zip(local_model.trainable_variables, updated_trainable_variables):
            local_var.assign(updated_var)

        vehicle_history = local_model.fit(training_input, training_output, batch_size=16, verbose=2).history
        vehicle_results.append({key: [list(map(int, vals))] for key, vals in vehicle_history.items()})
        vehicle_socket.send(pickle.dumps(local_model.trainable_variables))

    vehicle_num = len([filename for filename in os.listdir('../results/eval/')
                       if f'federated-{num_vehicles}-{model_name}' in filename])
    with open(f'../results/eval/federated-{num_vehicles}-{model_name}-vehicle-{vehicle_num}.json', 'w') as file:
        json.dump(vehicle_results, file)
    vehicle_socket.send('complete'.encode('utf8'))
    vehicle_socket.close()
