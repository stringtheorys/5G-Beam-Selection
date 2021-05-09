"""
Implementation of federated learning for beam alignment agents
"""

import json
import os
from typing import Callable

import numpy as np
import tensorflow as tf

from core.metrics import TopKThroughputRatio, top_k_metrics


def federated_training(name: str, model_fn: Callable[[], tf.keras.models.Model], num_vehicles: int,
                       training_input, validation_input, training_output, validation_output, epochs=30):
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
    """
    # The global and vehicle models
    global_model = model_fn()
    vehicle_models = [model_fn() for _ in range(num_vehicles)]
    [model.set_weights(global_model.get_weights()) for model in vehicle_models]

    # Metrics
    metrics = [
        tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top-1-accuracy'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top-10-accuracy'),
        TopKThroughputRatio(k=1, name='top-1-throughput'),
        TopKThroughputRatio(k=10, name='top-10-throughput')
    ]

    # Compile the models
    global_model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=metrics)
    for vehicle_model in vehicle_models:
        vehicle_model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=metrics)

    # Determine the training datasets
    if isinstance(training_input, tuple):
        training_input = [np.array_split(inputs, num_vehicles) for inputs in training_input]
        vehicle_training_dataset = list(zip(zip(*training_input), np.array_split(training_output, num_vehicles)))
    else:
        vehicle_training_dataset = list(zip(np.array_split(training_input, num_vehicles),
                                            np.array_split(training_output, num_vehicles)))

    # Save the metric history over training steps
    history = {model_name: {metric_name: [] for metric_name in ['loss', 'val_loss'] + [m.name for m in metrics] +
                            [f'val_{m.name}' for m in metrics]}
               for model_name in ['global'] + [f'vehicle {pos}' for pos in range(num_vehicles)]}
    for epochs in range(epochs):
        print(f'Epochs: {epochs}')
        for vehicle_num, (vehicle_model, training_data) in enumerate(zip(vehicle_models, vehicle_training_dataset)):
            vehicle_train_eval = vehicle_model.fit(*training_data, batch_size=16, verbose=2,
                                                   validation_data=(validation_input, validation_output)).history
            for metric_name, value in vehicle_train_eval.items():
                history[f'vehicle {vehicle_num}'][metric_name].append(value)

        # Add the each of the vehicle results to the global model
        vehicle_variables = [model.trainable_variables for model in vehicle_models]
        for global_weight, *vehicle_weights in zip(global_model.trainable_variables, *vehicle_variables):
            global_weight.assign(sum(weight for weight in vehicle_weights) / num_vehicles)

        # Update all of the vehicle models to copy the global model
        [model.set_weights(global_model.get_weights()) for model in vehicle_models]

        # Validation of the global model
        global_eval = global_model.evaluate(validation_input, validation_output, verbose=2, return_dict=True)
        for metric_name, value in global_eval.items():
            history['global'][metric_name].append(value)

    # Save all of the models
    if os.path.exists(f'../results/models/federated-{num_vehicles}-{name}'):
        os.remove(f'../results/models/federated-{num_vehicles}-{name}')
    os.mkdir(f'../results/models/federated-{num_vehicles}-{name}')
    for vehicle_num, vehicle_model in enumerate(vehicle_models):
        vehicle_model.save(f'../results/models/federated-{num_vehicles}-{name}/vehicle-{vehicle_num}-model/model')
    global_model.save(f'../results/models/federated-{num_vehicles}-{name}/global-model/model')

    # Top K metrics
    top_k_accuracy, top_k_throughput_ratio = top_k_metrics(global_model, validation_input, validation_output)
    with open(f'../results/eval/federated-{num_vehicles}-{name}.json', 'w') as file:
        json.dump({'top-k-accuracy': top_k_accuracy, 'top-k-throughput-ratio': top_k_throughput_ratio,
                   'history': history}, file)
