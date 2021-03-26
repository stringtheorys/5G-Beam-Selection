"""
Testing file for training of centralised beam alignment agent
"""

import argparse
import datetime
import json

import tensorflow as tf

from core.common import model_top_metric_eval, parse_args, TopKThroughputRatio

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='imperial', choices=['imperial', 'beamsoup-lidar',
                                                                  'beamsoup-coord', 'beamsoup'])


def centralised_training(name, model, training_input, training_output, validation_input, validation_output, epochs=15):
    # Loss and optimiser for the model
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimiser = tf.keras.optimizers.Adam()

    # The accuracy and throughput metrics
    top_1_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top-1-accuracy')
    top_10_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top-10-accuracy')
    top_1_throughput = TopKThroughputRatio(k=1, name='top-1-throughput')
    top_10_throughput = TopKThroughputRatio(k=10, name='top-10-throughput')

    # Adds callbacks over the epochs (through this is saved in the eval.json file)
    log_dir = f'../results/logs/{name}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Compile the model with the optimiser, loss and metrics
    model.compile(optimizer=optimiser, loss=loss, metrics=[top_1_acc, top_10_acc, top_1_throughput, top_10_throughput])

    # Train the model, change the epochs value for the number of training rounds
    history = model.fit(x=training_input, y=training_output, batch_size=16,  callbacks=tensorboard_callback,
                        validation_data=(validation_input, validation_output), epochs=epochs)
    model.save_weights(f'../results/models/centralised-{name}/model')

    # Custom evaluation of the trained model
    correct, top_k, throughput_ratio_k = model_top_metric_eval(model, validation_input, validation_output)
    # print(correct, top_k, throughput_ratio_k)
    with open(f'../results/centralised-{name}-eval.json', 'w') as file:
        json.dump({'correct': int(correct), 'top-k': top_k, 'throughput-ratio-k': throughput_ratio_k,
                   'history': {key: [int(val) for val in vals] for key, vals in history.history.items()}}, file)


if __name__ == '__main__':
    args, centralised_model, *train_validation_data = parse_args(parser)

    centralised_training(args.model, centralised_model, *train_validation_data)
