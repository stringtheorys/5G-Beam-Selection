"""
Base station example script for simulating the sending across of model weights and for clients to train updates
    these updates are returned to the base station

This script allows for three different training methods: centralised, distributed and federated along with the
    use of all models
"""

import argparse
import socket
from pickle import dumps, loads

import tensorflow as tf


# Script argument parser with a range of argument types: training, model, epochs, host, port and batch-size
from models import models

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--agent')  # Ignore
parser.add_argument('-t', '--training', default='distributed', choices=['centralised', 'distributed', 'federated'])
parser.add_argument('-m', '--model', default='imperial')  # Add choices
parser.add_argument('-e', '--epochs', default=1000)
parser.add_argument('-z', '--host', default='127.0.0.1')
parser.add_argument('-p', '--port', default=65432)
parser.add_argument('-b', '--batch-size', default=16)


def start(args):
    # Starts the server for the base station with the (host, port) address
    print(f'Starting base station server with address: ({args.host}, {args.port})')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as base_station_socket:
        base_station_socket.bind((args.host, args.port))

        print('[+] Base station is listening')
        base_station_socket.listen()  # Listen for the client to connect

        conn, addr = base_station_socket.accept()  # When client has connected with address
        print(f'[+] Connection accepted - {addr}')

        # Parse the selected model, determine the model size and initialise the optimiser
        model_fn, dataset_fn = models[args.model]
        model = model_fn()
        model_size = len(dumps(model.get_weights()))
        optimiser = tf.keras.optimizers.Adam()

        # Generate the training dataset for the centralised training method
        dataset = tf.data.Dataset.from_tensor_slices(tf.range(0, len(dataset_fn()[0])))
        dataset = dataset.shuffle(args.batch_size).repeat(args.epoches).batch(args.batch_size)
        dataset = dataset.make_one_shot_iterator()

        # Send the client the training and model information
        conn.send(f'training: {args.training}, model: {args.model}'.encode('ascii'))

        conn.recv(1024)  # Client returns 'ready' then completed its preprocessing
        for epoch in range(args.epochs):
            conn.send(dumps(model.get_weights()))  # Sends the client the current weights of the model

            if args.training == 'centralised':
                conn.send(dumps(dataset.get_next()))  # Sends the client, the dataset samples in which to use
                # Receives the model gradients from the client and update the model using the optimiser
                model_gradients = loads(base_station_socket.recv(model_size))
                optimiser.apply_gradients(zip(model_gradients, model.trainable_variables))
            elif args.training == 'distributed':
                # Receives the model gradients from the client and update the model using the optimiser
                model_gradients = loads(base_station_socket.recv(model_size))
                optimiser.apply_gradients(zip(model_gradients, model.trainable_variables))
            elif args.training == 'federated':
                # Receive the model weights from the client
                model_weights = loads(base_station_socket.recv(model_size))
                model.set_weights(model_weights)
            else:
                raise Exception(f'Unknown training type: {args.training}')

        # Send an empty msg to the client to indicate the training is over
        conn.send(b'')
