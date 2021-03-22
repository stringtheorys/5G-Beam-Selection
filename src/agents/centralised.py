"""
Testing file for training of centralised beam alignment agent
"""

import argparse
import json

import tensorflow as tf

from core.common import model_top_metric_eval, parse_args

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='imperial', choices=['imperial', 'bs-lidar', 'bs-coord', 'beamsoup'])


def centralised_training(name, model, training_input, training_output, validation_input, validation_output, epochs=20):
    # Loss, optimiser and metrics for the model
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimiser = tf.keras.optimizers.Adam()

    top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top-1')
    top10 = tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top-10')

    # Compile the model with the optimiser, loss and metrics
    model.compile(optimizer=optimiser, loss=loss, metrics=[top1, top10])

    # Tensorboard for logging of training info
    # log_directory = f'../logs/centralised-{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_directory)

    # Train the model, change the epochs value for the number of training rounds
    history = model.fit(x=training_input, y=training_output, batch_size=16,  # callbacks=tensorboard_callback,
                        validation_data=(validation_input, validation_output), epochs=epochs)
    model.save_weights(f'../models/centralised-{name}-model')

    # Custom evaluation of the trained model
    correct, top_k, throughput_ratio_k = model_top_metric_eval(model, validation_input, validation_output)
    print(correct, top_k, throughput_ratio_k)
    with open(f'../centralised-{model}-eval.json', 'w') as file:
        json.dump({'correct': int(correct), 'top-k': top_k, 'throughput-ratio-k': throughput_ratio_k,
                   'history': {key: [int(val) for val in vals] for key, vals in history.history.items()}}, file)


if __name__ == '__main__':
    args, centralised_model, *train_validation_data = parse_args(parser)

    centralised_training(args.model, centralised_model, *train_validation_data)
