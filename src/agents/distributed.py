"""
Testing file for training of centralised beam alignment agent
"""

import json

import tensorflow as tf
from tqdm import tqdm

from core.common import model_top_metric_eval, TopKThroughputRatio


@tf.function
def training_step(model, x, y, loss_fn, optimiser, metrics):
    """
    Training step

    :param model: keras model
    :param x: training input
    :param y: training output
    :param loss_fn: categorical loss function
    :param optimiser: keras optimiser
    :param metrics: dictionary of metrics
    :return: the error from the loss function
    """
    # Model prediction
    with tf.GradientTape() as tape:
        predicted = model(x, training=True)
        error = loss_fn(predicted, y)

    # Backpropagation
    gradients = tape.gradient(error, model.trainable_variables)
    optimiser.apply_gradients(zip(gradients, model.trainable_variables))

    # Update the metrics for the training
    for metric in metrics:
        metric.update_state(y, predicted)

    return error


@tf.function
def validation_step(model, x, y, metrics):
    predicted = model(x, training=False)

    for metric in metrics:
        metric.update_state(y, predicted)


def distributed_training(name, model, training_input, training_output, validation_input, validation_output,
                         epochs=15, batch_size=16, eval_steps=500):
    """
    Custom distributed training

    :param name: Model name
    :param model: tensorflow model
    :param training_input: training input dataset
    :param training_output: training output dataset
    :param validation_input: validation input dataset
    :param validation_output: validation output dataset
    :param epochs: number of epochs
    :param batch_size: batch training size
    :param eval_steps: number of training steps between evaluations of the model
    """
    # Loss and optimiser for the model
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimiser = tf.keras.optimizers.Adam()

    # Training and validation dataset
    training_dataset = tf.data.Dataset.from_tensor_slices((training_input, training_output))
    training_dataset = training_dataset.repeat(epochs).shuffle(buffer_size=1024).batch(batch_size)

    # List of metrics for the model
    metrics = [
        tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top-1-accuracy'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top-10-accuracy'),
        TopKThroughputRatio(k=1, name='top-1-throughput'),
        TopKThroughputRatio(k=10, name='top-10-throughput')
    ]

    # Saves the metric history over training steps
    history = []
    for step, (training_x, training_y) in tqdm(enumerate(training_dataset)):
        error = training_step(model, training_x, training_y, loss_fn, optimiser, metrics)

        # Every eval_steps then calculate all of the metrics and save them
        if step % eval_steps == 0:
            # Training evaluation
            metric_results = {'loss': float(error.numpy())}
            for metric in metrics:
                metric_results[f'training-{metric.name}'] = float(metric.result().numpy())
                metric.reset_states()

            # Validation evaluation
            validation_step(model, validation_input, validation_output, metrics)
            for metric in metrics:
                metric_results[f'validation-{metric.name}'] = float(metric.result().numpy())
                metric.reset_states()

            print(f'step: {step} - {metric_results}')
            history.append(metric_results)
        model.save_weights(f'../results/models/distributed-{name}/model')

    # Custom evaluation of the trained model
    print('Logging the evaluations')
    correct, top_k, throughput_ratio_k = model_top_metric_eval(model, validation_input, validation_output)
    with open(f'../results/distributed-{name}-eval.json', 'w') as file:
        json.dump({'correct': int(correct), 'top-k': top_k, 'throughput-ratio-k': throughput_ratio_k,
                   'history': history}, file)
