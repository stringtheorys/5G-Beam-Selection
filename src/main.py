import argparse
import os

from agents.centralised import centralised_training
from agents.federated import federated_training
from communication import basestation, vehicle
from core.dataset import output_dataset
from models import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--agent', default='centralised',
                    choices=['centralised', 'centralised-v2', 'federated', 'basestation', 'vehicle'])
parser.add_argument('-m', '--model', default='imperial', choices=models.keys())
parser.add_argument('-v', '--vehicles', default=2)

if __name__ == '__main__':
    args = parser.parse_args()
    print(f'Agents: {args.agent}, Model: {args.model}')

    if args.agent == 'basestation':
        basestation.start()
    elif args.agent == 'vehicle':
        vehicle.start()
    else:
        model_fn, dataset_fn = models[args.model]

        if args.agent == 'centralised':
            centralised_training(args.model, model_fn(), *dataset_fn(), *output_dataset())
        elif args.agent == 'centralised-v2':
            centralised_training(f'{args.model}-v2', model_fn(), *dataset_fn(), *output_dataset(version='v2'))
        elif args.agent == 'federated':
            federated_training(args.model, model_fn, int(args.vehicles), *dataset_fn(), *output_dataset())
        else:
            raise Exception(f'Unknown agent type: {args.agent}')
