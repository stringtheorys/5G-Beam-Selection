
import gc

from agents.centralised import centralised_training
from agents.distributed import distributed_training
from agents.federated import federated_training
from communication import basestation, jetson
from core.io import parse_model, parser

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
        model_fn, *train_validation_data = parse_model(args.model)
        gc.collect()

        if args.agent == 'centralised':
            centralised_training(args.model, model_fn(), *train_validation_data)
        elif args.agent == 'distributed':
            distributed_training(args.model, model_fn(), *train_validation_data)
        elif args.agent == 'federated':
            federated_training(args.model, model_fn, *train_validation_data)
        elif args.agent == 'southampton':
            model_fn, *train_validation_data = parse_model(args.model, version='v2')
            gc.collect()

            centralised_training(args.model + '_v2', model_fn(), *train_validation_data)
        else:
            raise Exception(f'Unknown agent type: {args.agent}')
