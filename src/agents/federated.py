import json
from typing import Callable

import tensorflow as tf
from tqdm import tqdm

from core.metrics import TopKThroughputRatio, top_k_metrics
from core.training import training_step, validation_step


def federated_training(name: str, model_fn: Callable[[], tf.keras.models.Model], num_vehicles: int,
                       training_input, validation_input, training_output, validation_output,
                       epochs=15, batch_size=16):
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
    :param batch_size: batch training size
    """
    print(f'Federated learning for {name}')
    # Loss and optimiser for the each vehicle model
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    global_optimiser = tf.keras.optimizers.SGD(lr=0.025)
    vehicle_optimisers = [tf.keras.optimizers.Adam() for _ in range(num_vehicles)]

    # Vehicle and global models
    global_model = model_fn()
    vehicle_models = [model_fn() for _ in range(num_vehicles)]

    # Generate the training datasets
    vehicle_training_dataset = tf.data.Dataset.from_tensor_slices((training_input, training_output))
    vehicle_training_dataset = tf.split(vehicle_training_dataset, num_split=num_vehicles)  # maybe change the axis=-1
    vehicle_training_dataset = [dataset.shuffle(buffer_size=512).batch(batch_size)
                                for dataset in vehicle_training_dataset]

    # Vehicle metrics
    loss_metric = tf.keras.metrics.Mean(name=f'loss')
    metrics = [
        tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top-1-accuracy'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top-10-accuracy'),
        TopKThroughputRatio(k=1, name='top-1-throughput'),
        TopKThroughputRatio(k=10, name='top-10-throughput')
    ]

    # Save the metric history over training steps
    history = []
    for epochs in tqdm(epochs):
        print(f'Epochs: {epochs}')
        epoch_results = {}
        for vehicle_id in range(num_vehicles):
            for training_x, training_y in vehicle_training_dataset[vehicle_id]:
                loss = training_step(vehicle_models[vehicle_id], training_x, training_y, loss_fn,
                                     vehicle_optimisers[vehicle_id], metrics)
                loss_metric[vehicle_id].update_state(loss)

            # Vehicle results
            vehicle_result = {'loss': float(loss_metric.result().numpy())}
            loss_metric.reset_states()

            # Training metrics for the vehicle
            for metric in metrics:
                vehicle_result[f'training-{metric.name}'] = float(metric.result().numpy())
                metric.reset_states()

            # Validation metrics for the vehicle
            validation_step(vehicle_models[vehicle_id], validation_input, validation_output, metrics)
            for metric in metrics:
                vehicle_result[f'validation-{metric.name}'] = float(metric.result().numpy())
                metric.reset_states()

            print(f'\tVehicle id: {vehicle_id} - {vehicle_result}')
            epoch_results[f'vehicle {vehicle_id}'] = vehicle_result

        # Add the each of the vehicle results to the global model
        avg_weights = tf.reduce_mean([model.get_weights() for model in vehicle_models])
        global_optimiser.apply_gradients(zip(global_model.trainable_variables, avg_weights))
        
        # Validation of the global model
        validation_step(global_model, validation_input, validation_output)
        global_results = {}
        for metric in metrics:
            global_results[f'validation-{metric.name}'] = float(metric.result().numpy())
            metric.reset_states()
        epoch_results['global'] = global_results

        # Add the epoch results to the history
        history.append(epoch_results)
    global_model.save_weights(f'../results/models/federated-{num_vehicles}-{name}/model')

    # Top K metrics
    top_k_accuracy, top_k_throughput_ratio = top_k_metrics(global_model, validation_input, validation_output)
    with open(f'../results/federated-{num_vehicles}-{name}-eval.json', 'w') as file:
        json.dump({'top-k-accuracy': top_k_accuracy, 'top-k-throughput-ratio': top_k_throughput_ratio,
                   'history': history}, file)

