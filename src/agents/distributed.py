"""
Testing file for training of centralised beam alignment agent
"""

import json

import tensorflow as tf
from tqdm import tqdm

from core.metrics import top_k_metrics, TopKThroughputRatio
from core.training import training_step, validation_step


def distributed_training(name, model, training_input, training_output, validation_input, validation_output,
                         epochs=15, batch_size=16):
    """
    Custom distributed training with a centralised dataset

    :param name: Model name
    :param model: tensorflow model
    :param training_input: training input dataset
    :param training_output: training output dataset
    :param validation_input: validation input dataset
    :param validation_output: validation output dataset
    :param epochs: number of epochs
    :param batch_size: batch training size
    """
    # Loss and optimiser for the model
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimiser = tf.keras.optimizers.Adam()

    # Training and validation dataset
    training_dataset = tf.data.Dataset.from_tensor_slices((training_input, training_output))
    training_dataset = training_dataset.repeat(epochs).shuffle(buffer_size=1024).batch(batch_size)

    # List of metrics for the model
    loss_metric = tf.keras.metrics.Mean(name='loss')
    metrics = [
        tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top-1-accuracy'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top-10-accuracy'),
        TopKThroughputRatio(k=1, name='top-1-throughput'),
        TopKThroughputRatio(k=10, name='top-10-throughput')
    ]

    # Saves the metric history over training steps
    history = []
    for epoch in range(epochs):
        # Training steps
        for training_x, training_y in tqdm(training_dataset):
            loss = training_step(model, training_x, training_y, loss_fn, optimiser, metrics)
            loss_metric.update_state(loss)

        # Epoch results
        epoch_results = {'loss': float(loss_metric.result().numpy())}
        loss_metric.reset_states()

        # Training metrics
        for metric in metrics:
            epoch_results[f'training-{metric.name}'] = float(metric.result().numpy())
            metric.reset_states()

        # Validation metrics
        validation_step(model, validation_input, validation_output, metrics)
        for metric in metrics:
            epoch_results[f'validation-{metric.name}'] = float(metric.result().numpy())
            metric.reset_states()

        print(f'Epoch: {epoch} - {epoch_results}')
        history.append(epoch_results)
    model.save_weights(f'../results/models/distributed-{name}/model')

    # Top K metrics
    top_k_accuracy, top_k_throughput_ratio = top_k_metrics(model, validation_input, validation_output)
    with open(f'../results/distributed-{name}-eval.json', 'w') as file:
        json.dump({'top-k-accuracy': top_k_accuracy, 'top-k-throughput-ratio': top_k_throughput_ratio,
                   'history': history}, file)
