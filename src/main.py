import argparse

import tensorflow as tf

from agents.centralised import centralised_training
from agents.distributed import distributed_training
from agents.federated import federated_training
from communication import basestation, jetson
from core.dataset import output_dataset
from models import models

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--agent', default='centralised',
                    choices=['centralised', 'distributed', 'federated', 'southampton', 'basestation', 'jetson'])
parser.add_argument('-m', '--model', default='imperial', choices=models.keys())
parser.add_argument('-v', '--vehicles', default=2)

if __name__ == '__main__':
    args = parser.parse_args()
    print(f'Agents: {args.agent}, Model: {args.model}')

    if args.agent == 'basestation':
        basestation.start(basestation.parser.parse_args())
    elif args.agent == 'jetson':
        jetson.start(jetson.parser.parse_args())
    else:
        model_fn, dataset_fn = models[args.model]

        if args.agent == 'centralised':
            centralised_training(args.model, model_fn(), *dataset_fn(), *output_dataset(), epochs=30)
        elif args.agent == 'distributed':
            distributed_training(args.model, model_fn(), *dataset_fn(), *output_dataset(), epochs=30)
        elif args.agent == 'federated':
            federated_training(args.model, model_fn, int(args.vehicles), *dataset_fn(), *output_dataset())
        elif args.agent == 'southampton':
            centralised_training(f'{args.model}-v2', model_fn(), *dataset_fn(), *output_dataset(version='v2'),
                                 loss_fn=tf.keras.losses.MeanSquaredError())
        else:
            raise Exception(f'Unknown agent type: {args.agent}')
