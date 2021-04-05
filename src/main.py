import argparse

from agents.centralised import centralised_training
from agents.distributed import distributed_training
from agents.federated import federated_training
from agents.southampton import southampton_training
from core.common import parse_model

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--agent', default='centralised', choices=['centralised', 'distributed', 'federated',
                                                                     'southampton'])
parser.add_argument('-m', '--model', default='imperial', choices=['imperial', 'beamsoup-lidar',
                                                                  'beamsoup-coord', 'beamsoup'])


if __name__ == '__main__':
    args = parser.parse_args()
    model_name, model, *train_validation_data = parse_model(parser)

    if args.agent == 'centralised':
        centralised_training(model_name, model, *train_validation_data)
    elif args.agent == 'distributed':
        distributed_training(model_name, model, *train_validation_data)
    elif args.agent == 'federated':
        federated_training()
    elif args.agent == 'southampton':
        southampton_training()
