"""
Training for a centralised version of the beam alignment agent
"""

import datetime as dt
import json
import os

import numpy as np
import tensorflow as tf

from core.metrics import top_k_metrics, TopKThroughputRatio


def centralised_training(name: str, model: tf.keras.models.Sequential,
                         training_input: np.ndarray, validation_input: np.ndarray,
                         training_output: np.ndarray, validation_output: np.ndarray, epochs: int = 30):
    """
    Centralised training agent

    :param name: model name
    :param model: tensorflow model
    :param training_input: numpy matrix for the training input
    :param validation_input: numpy matrix for the validation input
    :param training_output: numpy matrix for the training output
    :param validation_output: numpy matrix for the validation output
    :param epochs: number of epochs the mode.fit function will run for
    """
    # The accuracy and throughput metrics
    metrics = [
        tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top-1-accuracy'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top-10-accuracy'),
        TopKThroughputRatio(k=1, name='top-1-throughput'),
        TopKThroughputRatio(k=10, name='top-10-throughput')
    ]

    # Adds callbacks over the epochs (through this is saved in the eval.json file)
    log_dir = f'../results/logs/centralised-{name}/{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False, profile_batch=2)

    # Compile the model with the optimiser, loss and metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=metrics)

    # Train the model, change the epochs value for the number of training rounds
    history = model.fit(x=training_input, y=training_output, batch_size=16, callbacks=tensorboard_callback,
                        validation_data=(validation_input, validation_output), epochs=epochs, verbose=2)

    # Save the model
    if os.path.exists(f'../results/models/centralised-{name}'):
        os.remove(f'../results/models/centralised-{name}')
    os.mkdir(f'../results/models/centralised-{name}')
    model.save(f'../results/models/centralised-{name}/model')

    # Top K metrics
    top_k_accuracy, top_k_throughput_ratio = top_k_metrics(model, validation_input, validation_output)
    with open(f'../results/eval/centralised-{name}.json', 'w') as file:
        json.dump({'top-k-accuracy': top_k_accuracy, 'top-k-throughput-ratio': top_k_throughput_ratio,
                   'history': {key: [list(map(int, values))] for key, values in history.history.items()}}, file)
