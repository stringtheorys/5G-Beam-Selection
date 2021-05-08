"""
Implementation of federated learning for beam alignment agents
"""

import datetime as dt
import json
import os
from typing import Callable

import numpy as np
import tensorflow as tf

from core.metrics import TopKThroughputRatio, top_k_metrics
from core.training import validation_step


def federated_training(name: str, model_fn: Callable[[], tf.keras.models.Model], num_vehicles: int,
                       training_input, validation_input, training_output, validation_output,
                       epochs=30, loss_fn: tf.keras.losses.Loss = tf.keras.losses.CategoricalCrossentropy()):
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
    global_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss_fn, metrics=metrics)
    for vehicle_model in vehicle_models:
        vehicle_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss_fn, metrics=metrics)

    # Determine the training datasets
    vehicle_training_dataset = list(zip(tuple(np.array_split(inputs, num_vehicles) for inputs in training_input),
                                        np.array_split(training_output, num_vehicles)))

    # Adds callbacks over the epochs (through this is saved in the eval.json file)
    log_dir = f'../results/logs/federated-{name}-{num_vehicles}/{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    global_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'{log_dir}/global', write_graph=True)
    vehicle_tensorboard_callback = [
        tf.keras.callbacks.TensorBoard(log_dir=f'{log_dir}/vehicle-{vehicle_id}', write_graph=False)
        for vehicle_id in range(num_vehicles)
    ]

    # Save the metric history over training steps
    history = []
    for epochs in range(epochs):
        print(f'Epochs: {epochs}')
        epoch_results = {}
        for vehicle_id, (vehicle_model, training_data) in enumerate(zip(vehicle_models, vehicle_training_dataset)):
            vehicle_history = vehicle_model.fit(*training_data, batch_size=16, verbose=2,
                                                validation_data=(validation_input, validation_output),
                                                callbacks=vehicle_tensorboard_callback[vehicle_id]).history
            epoch_results[f'vehicle {vehicle_id}'] = {key: [list(map(int, vals))]
                                                      for key, vals in vehicle_history.items()}

        # Add the each of the vehicle results to the global model
        vehicle_variables = [model.trainable_variables for model in vehicle_models]
        for global_weight, *vehicle_weights in zip(global_model.trainable_variables, *vehicle_variables):
            global_weight.assign(sum(weight for weight in vehicle_weights) / num_vehicles)

        # Validation of the global model
        global_evaluation = global_model.evaluate(validation_input, validation_output, verbose=2,
                                                  callbacks=[global_tensorboard_callback])
        epoch_results['global'] = dict(zip(['loss'] + [m.name for m in metrics], global_evaluation))

        # Add the epoch results to the history
        history.append(epoch_results)

        # Update all of the vehicle models to copy the global model
        [model.set_weights(global_model.get_weights()) for model in vehicle_models]

    # Save all of the models
    if os.path.exists(f'../results/models/federated-{num_vehicles}-{name}'):
        os.remove(f'../results/models/federated-{num_vehicles}-{name}')
    os.mkdir(f'../results/models/federated-{num_vehicles}-{name}')
    for vehicle_id, vehicle_model in enumerate(vehicle_models):
        vehicle_model.save(f'../results/models/federated-{num_vehicles}-{name}/vehicle-{vehicle_id}-model/model')
    global_model.save(f'../results/models/federated-{num_vehicles}-{name}/global-model/model')

    # Top K metrics
    top_k_accuracy, top_k_throughput_ratio = top_k_metrics(global_model, validation_input, validation_output)
    with open(f'../results/eval/federated-{num_vehicles}-{name}.json', 'w') as file:
        json.dump({'top-k-accuracy': top_k_accuracy, 'top-k-throughput-ratio': top_k_throughput_ratio,
                   'history': history}, file)
