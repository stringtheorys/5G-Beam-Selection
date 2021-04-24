
import argparse
import gc

from agents.centralised import centralised_training
from agents.distributed import distributed_training
# from agents.federated import federated_training
from agents.southampton import southampton_training
from communication import basestation, jetson
from core.common import parse_model

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--agent', default='centralised', choices=['centralised', 'distributed', # 'federated',
                                                                     'southampton', 'basestation', 'jetson'])
model_choices = ['imperial', 'beamsoup-coord', 'beamsoup-lidar', 'beamsoup-joint',
                 'husky-coord', 'husky-lidar', 'husky-image', 'husky-fusion',
                 'baseline-coord', 'baseline-lidar', 'baseline-image', 'baseline-fusion']
parser.add_argument('-m', '--model', default='imperial', choices=model_choices)


if __name__ == '__main__':
    args = parser.parse_args()
    print(f'Agents: {args.agent}, Model: {args.model}')

    if args.agent == 'basestation':
        parsed_args = basestation.parser.parse_args()

        basestation.start(parsed_args)
    elif args.agent == 'jetson':
        parsed_arg = jetson.parser.parse_args()

        jetson.start(parsed_arg)
    else:
        model, *train_validation_data = parse_model(args.model)
        gc.collect()

        if args.agent == 'centralised':
            centralised_training(args.model, model, *train_validation_data)
        elif args.agent == 'distributed':
            distributed_training(args.model, model, *train_validation_data)
        # elif args.agent == 'federated':
        #     federated_training()
        elif args.agent == 'southampton':
            southampton_training()
        else:
            raise Exception(f'Unknown agent type: {args.agent}')
