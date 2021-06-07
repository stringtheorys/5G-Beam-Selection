"""
Implementation of a 5G basestation
"""

import json
import pickle
import socket
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from core.dataset import output_dataset
from core.metrics import TopKThroughputRatio, top_k_metrics
from core.training import validation_step
from models import models


def start(model_name='imperial', num_vehicles=1, epochs=20, model_dtype=np.float32):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as basestation_socket:
        basestation_socket.bind(('169.254.176.117', 137))

        basestation_socket.listen(num_vehicles)
        vehicle_sockets = []

        model_fn, dataset_fn = models[model_name]
        global_model = model_fn()
        model_recv_size = len(pickle.dumps(global_model.trainable_variables))

        _, validation_input = dataset_fn()
        _, validation_output = output_dataset()

        # Metrics
        metrics = [
            tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top-1-accuracy'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top-10-accuracy'),
            TopKThroughputRatio(k=1, name='top-1-throughput'),
            TopKThroughputRatio(k=10, name='top-10-throughput')
        ]

        plt.ion()
        fig, axs = plt.subplots(2, 2, figsize=(14, 7))
        for ax, metric in zip(axs.flatten(), metrics):
            ax.set_title(metric.name)
            ax.set_xlim(.5, epochs + .5)
            ax.set_xlabel('Epoch')
        plt.tight_layout()
        plt.show()

        print('Listening for connections')
        for _ in range(num_vehicles):
            vehicle_socket, addr = basestation_socket.accept()

            print('Successful connection')
            vehicle_socket.send(model_name.encode('utf8'))
            vehicle_sockets.append(vehicle_socket)

        global_results = []
        for epoch in range(epochs):
            print(f'Epoch: {epoch}')

            # Update each of the vehicles with the updated model and receive their updated model
            vehicle_variables = []
            for pos, vehicle_socket in enumerate(vehicle_sockets):
                print(f'\tVehicle training: {pos}')
                vehicle_socket.send(pickle.dumps(global_model.trainable_variables))
                print('sent_w_size:', len(pickle.dumps(global_model.trainable_variables)))
                received_weights = vehicle_socket.recv(34429)
                print('rec_w_size:', len(received_weights))
                updated_vehicle_variables = pickle.loads(received_weights)
                vehicle_variables.append(updated_vehicle_variables)
            # Update the global model with the local vehicle models
            for global_weight, *vehicle_weights in zip(global_model.trainable_variables, *vehicle_variables):
                global_weight.assign(sum(1 / num_vehicles * weight for weight in vehicle_weights))

            # Validation of the global model
            validation_step(global_model, validation_input, validation_output, metrics)
            epoch_results = {}
            for metric in metrics:
                epoch_results[f'validation-{metric.name}'] = float(metric.result().numpy())
                metric.reset_states()
            global_results.append(epoch_results)
            print(f'Global model results: {epoch_results}')

            # Plots the results
            for ax, metric in zip(axs.flatten(), metrics):
                ax.plot(np.arange(epoch + 1) + 1, [result[f'validation-{metric.name}'] for result in global_results],
                        label='validation' if epoch == 0 else '', color='orange')
                ax.legend()

            plt.draw()
            plt.pause(0.01)

        # Send an empty string to the vehicle to inform them it has ended
        for vehicle_socket in vehicle_sockets:
            vehicle_socket.send('stop'.encode('utf8'))
            vehicle_socket.recv(256).decode('utf8')
            vehicle_socket.close()

        # Save the global model
        global_model.save(f'../results/models/federated-{num_vehicles}-{model_name}/global-model/model')

        # Top K metrics
        top_k_accuracy, top_k_throughput_ratio = top_k_metrics(global_model, validation_input, validation_output)
        with open(f'../results/eval/federated-{num_vehicles}-{model_name}.json', 'w') as file:
            json.dump({'top-k-accuracy': top_k_accuracy, 'top-k-throughput-ratio': top_k_throughput_ratio,
                       'history': global_results}, file)

        time.sleep(5)
