"""
Testing file for training of centralised beam alignment agent
"""

import argparse
import json

import numpy as np
import tensorflow as tf

from core.common import model_top_metric_eval, parse_args

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='imperial', choices=['imperial', 'bs-lidar', 'bs-coord', 'beamsoup'])


def distributed_training(name, model, training_input, training_output, validation_input, validation_output,
                         training_steps=1000, batch_size=16, eval_steps=20):
    # Loss, optimiser and metrics for the model
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimiser = tf.keras.optimizers.Adam()

    top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top-1')
    top10 = tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top-10')

    # Tensorboard for logging of training info
    # log_directory = f'../logs/centralised-{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_directory)

    history = []
    for step in range(training_steps):
        sample_indexes = np.random.uniform(0, len(training_input), batch_size)

        with tf.GradientTape() as tape:
            predict = model(training_input[sample_indexes])
            error = loss(predict, training_output[sample_indexes])

        gradients = tape.gradients(error, model.trainable_variables)
        optimiser.apply_gradients(zip(gradients, model.trainable_variables))

        if step % eval_steps == 0:
            sample_indexes = np.random.uniform(0, len(training_input), 128)
            history.append({
                'training-top-1': top1(training_input[sample_indexes], training_output[sample_indexes]),
                'training-top-10': top10(training_input[sample_indexes], training_output[sample_indexes]),
                'validation-top-1': top1(validation_input, validation_output),
                'validation-top-10': top10(validation_input, validation_output)
            })
        model.save_weights(f'../models/distributed-{name}-model')

    # Custom evaluation of the trained model
    correct, top_k, throughput_ratio_k = model_top_metric_eval(model, validation_input, validation_output)
    print(correct, top_k, throughput_ratio_k)
    with open(f'../distributed-{model}-eval.json', 'w') as file:
        json.dump({'correct': int(correct), 'top-k': top_k, 'throughput-ratio-k': throughput_ratio_k,
                   'history': history}, file)


if __name__ == '__main__':
    args, distributed_model, *train_validation_data = parse_args(parser)

    distributed_training(args.model, distributed_model, *train_validation_data)
