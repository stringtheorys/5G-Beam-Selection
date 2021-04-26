import json
import os
from typing import Callable

import tensorflow as tf
from tqdm import tqdm
import numpy as np

from core.metrics import TopKThroughputRatio, top_k_metrics
from core.training import validation_step


def federated_training(name: str, model_fn: Callable[[], tf.keras.models.Model], num_vehicles: int,
                       training_input, validation_input, training_output, validation_output,
                       epochs=15, loss_fn: tf.keras.losses.Loss = tf.keras.losses.CategoricalCrossentropy()):
    """
    Custom federated training

    :param name: Model name
    :param model_fn: function for creating a tensorflow model
    :param num_vehicles: the number of vehicles
    :param training_input: training input dataset
    :param training_output: training output dataset
    :param validation_input: validation input dataset
    :param validation_output: validation output dataset
    :param epochs: number of epochs
    :param loss_fn: the loss function
    """
    # The global model and optimiser
    global_model = model_fn()
    global_optimiser = tf.keras.optimizers.SGD(lr=0.025)

    # Vehicle and global models
    vehicle_models = [model_fn() for _ in range(num_vehicles)]
    metrics = [
        tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top-1-accuracy'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top-10-accuracy'),
        TopKThroughputRatio(k=1, name='top-1-throughput'),
        TopKThroughputRatio(k=10, name='top-10-throughput')
    ]
    for vehicle_model in vehicle_models:
        vehicle_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss_fn, metrics=metrics)

    # Generate the training datasets
    vehicle_training_dataset = [
        (inputs, outputs) for inputs, outputs in zip(tf.split(training_input, 3), tf.split(training_output, 3))
    ]

    # Save the metric history over training steps
    history = []
    for epochs in tqdm(range(epochs)):
        print(f'Epochs: {epochs}')
        epoch_results = {}
        for vehicle_id, vehicle_model in enumerate(vehicle_models):
            vehicle_history = vehicle_model.fit(*vehicle_training_dataset[vehicle_id], batch_size=16, verbose=2,
                                                validation_data=(validation_input, validation_output)).history

            print(f'\tVehicle id: {vehicle_id} - {vehicle_history}')
            epoch_results[f'vehicle {vehicle_id}'] = {key: [map(int, vals)] for key, vals in vehicle_history.items()}
            [metric.reset_states() for metric in metrics]

        # Add the each of the vehicle results to the global model
        vehicle_weights = [model.get_weights() for model in vehicle_models]
        avg_weights = [[np.array(weights).mean(axis=0) for weights in zip(*vehicle_layer)]
                       for vehicle_layer in zip(*vehicle_weights)]
        global_optimiser.apply_gradients(zip(avg_weights, global_model.trainable_variables))
        
        # Validation of the global model
        validation_step(global_model, validation_input, validation_output)
        global_results = {}
        for metric in metrics:
            global_results[f'validation-{metric.name}'] = float(metric.result().numpy())
            metric.reset_states()
        epoch_results['global'] = global_results

        # Add the epoch results to the history
        history.append(epoch_results)

    # Save all of the models
    if os.path.exists(f'../results/models/federated-{num_vehicles}-{name}'):
        os.remove(f'../results/models/federated-{num_vehicles}-{name}')
    os.mkdir(f'../results/models/federated-{num_vehicles}-{name}')
    for vehicle_id, vehicle_model in enumerate(vehicle_models):
        vehicle_model.save(f'../results/models/federated-{num_vehicles}-{name}/vehicle-{vehicle_id}-model/model')
    global_model.save(f'../results/models/federated-{num_vehicles}-{name}/global-model/model')

    # Top K metrics
    top_k_accuracy, top_k_throughput_ratio = top_k_metrics(global_model, validation_input, validation_output)
    with open(f'../results/federated-{num_vehicles}-{name}-eval.json', 'w') as file:
        json.dump({'top-k-accuracy': top_k_accuracy, 'top-k-throughput-ratio': top_k_throughput_ratio,
                   'history': history}, file)
