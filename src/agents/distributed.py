"""
Testing file for training of centralised beam alignment agent
"""

import json

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from core.common import model_top_metric_eval, TopKThroughputRatio


def distributed_training(name, model, training_input, training_output, validation_input, validation_output,
                         training_steps=5000, batch_size=16, eval_steps=500):
    """
    Custom distributed training

    :param name: Model name
    :param model: tensorflow model
    :param training_input: training input dataset
    :param training_output: training output dataset
    :param validation_input: validation input dataset
    :param validation_output: validation output dataset
    :param training_steps: the number of training steps
    :param batch_size: batch training size
    :param eval_steps: number of training steps between evaluations of the model
    """
    # Loss and optimiser for the model
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimiser = tf.keras.optimizers.Adam()

    # List of metrics for the model
    metrics = [
        tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top-1-accuracy'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top-10-accuracy'),
        TopKThroughputRatio(k=1, name='top-1-throughput'),
        TopKThroughputRatio(k=10, name='top-10-throughput')
    ]

    # Saves the metric history over training steps
    history = []
    for step in tqdm(range(training_steps)):
        # Collect a batch size number of random samples with uniform distribution
        with tf.GradientTape() as tape:
            sample_indexes = np.random.randint(0, len(training_input), batch_size)

            # Calculate the loss between the predicted output and the actual output
            error = loss(model(tf.gather(training_input, sample_indexes)), tf.gather(training_output, sample_indexes))

        # Calculate the gradients of the weights for trainable variables
        gradients = tape.gradient(error, model.trainable_variables)
        optimiser.apply_gradients(zip(gradients, model.trainable_variables))

        # Every eval_steps then calculate all of the metrics and save them
        if step % eval_steps == 0:
            indexes = np.random.randint(0, len(training_input), 128)
            sample_input, sample_output = tf.gather(training_input, indexes), tf.gather(training_output, indexes)

            metric_results = {}
            for metric in metrics:
                metric.update_state(model(sample_input), sample_output)
                metric_results[f'training-{metric.name}'] = metric.result().numpy()
                metric.update_state(model(validation_input), validation_output)
                metric_results[f'validation-{metric.name}'] = metric.result().numpy()

            history.append(metric_results)
            print(f'step: {step} - {metric_results}')
        model.save_weights(f'../results/models/distributed-{name}/model')

    # Custom evaluation of the trained model
    correct, top_k, throughput_ratio_k = model_top_metric_eval(model, validation_input, validation_output)
    # print(correct, top_k, throughput_ratio_k)
    with open(f'../results/distributed-{model}-eval.json', 'w') as file:
        json.dump({'correct': int(correct), 'top-k': top_k, 'throughput-ratio-k': throughput_ratio_k,
                   'history': history}, file)

