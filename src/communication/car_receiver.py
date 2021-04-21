"""
Car receiver example script for simulating the sending across of model weights and for clients to train updates
    these updates are returned to the base station
"""

import argparse
import socket
from pickle import dumps, loads

import tensorflow as tf

from core.common import parse_model

# Script argument parser with the server host and port
parser = argparse.ArgumentParser()
parser.add_argument('-h', '--host', default='127.0.0.1')
parser.add_argument('-p', '--port', default=65432)
parser.add_argument('-b', '--batch-size', default=16)


@tf.function
def training_step(model, x, y, loss_fn):
    with tf.GradientTape() as tape:
        prediction = model(x)
        loss = loss_fn(prediction, y)

    gradients = tape.gradient(loss, model.trainable_variables)
    return loss, gradients


def start(args):
    # Starts the client for the car receiver with the (host, port) address
    print(f'Starting car receiver; connecting to: ({args.host}, {args.port})')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as receiver_socket:
        receiver_socket.connect((args.host, args.port))

        # Receive training and model type
        data = receiver_socket.recv(1024).decode('ascii').split(', ')
        training_type, model_type = data[0].replace('training: ', ''), data[1].replace('model: ', '')
        print(f'Using training type: {training_type} and model type: {model_type}')

        # Parse and generate the model then tell the server that it is ready to start training
        model, training_input, training_output, validation_input, validation_output = parse_model(model_type)
        model_size = len(dumps(model.get_weights()))
        samples_size = len(dumps(tf.range(args.batch_size)))
        optimiser = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.CategoricalCrossentropy()

        # When the client is ready will send over all of the
        receiver_socket.send(b'ready')

        while True:
            data = receiver_socket.recv(model_size)
            if not data:
                print('Ending training')
                break

            # Receives and decodes the model trainable weights
            model.set_weights(loads(data))
            # Train using the three types of learning methods
            if training_type == 'centralised':
                training_indexes = loads(receiver_socket.recv(samples_size))

                gradients, loss = training_step(model, tf.gather(training_input, training_indexes),
                                                tf.gather(training_output, training_indexes), loss_fn)
                receiver_socket.send(dumps(gradients))
            else:
                training_indexes = tf.random.uniform(args.batch_size, 0, len(training_input), dtype=tf.int32)
                gradients, loss = training_step(model, tf.gather(training_input, training_indexes),
                                                tf.gather(training_output, training_indexes), loss_fn)
                if training_type == 'distributed':
                    receiver_socket.send(dumps(gradients))
                elif training_type == 'federated':
                    optimiser.apply_gradients(zip(gradients, model.trainable_variables))
                    receiver_socket.send(dumps(model.get_weights()))
                else:
                    raise Exception(f'Unknown training type: {training_type}')
