import argparse
from typing import Tuple

import numpy as np
import tensorflow as tf

from core.dataset import lidar_to_2d, beam_outputs, beam_outputs_v2
from models.baseline import baseline_coord_model_fn, baseline_lidar_model_fn, baseline_image_model_fn, \
    baseline_fusion_model_fn
from models.beamsoup import beamsoup_coord_model_fn, beamsoup_lidar_model_fn, beamsoup_joint_model_fn
from models.imperial import imperial_model_fn
from models.nu_huskies import nu_husky_coord_model_fn, nu_husky_lidar_model_fn, nu_husky_image_model_fn, \
    nu_husky_fusion_model_fn
from models.southampton import southampton_model_fn


parser = argparse.ArgumentParser()
parser.add_argument('-a', '--agent', default='centralised', choices=['centralised', 'distributed', 'federated',
                                                                     'southampton', 'basestation', 'jetson'])
model_choices = ['imperial', 'beamsoup-coord', 'beamsoup-lidar', 'beamsoup-joint',
                 'husky-coord', 'husky-lidar', 'husky-image', 'husky-fusion',
                 'baseline-coord', 'baseline-lidar', 'baseline-image', 'baseline-fusion']
parser.add_argument('-m', '--model', default='imperial', choices=model_choices)
parser.add_argument('-v', '--vehicle', default=2)


def parse_model(model_name: str, folder: str = '../data', version: str = 'v1') \
        -> Tuple[tf.keras.models.Sequential, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parsing the model name and training/validation data

    :param model_name: The model name
    :param folder: location of the data folder
    :param version: Version of the beam output
    :return: tensorflow model, training input and output, validation input and output
    """
    # Load the training and validation datasets
    train_lidar_data = np.transpose(np.expand_dims(lidar_to_2d(f'{folder}/lidar_train.npz'), 1), (0, 2, 3, 1))
    val_lidar_data = np.transpose(np.expand_dims(lidar_to_2d(f'{folder}/lidar_validation.npz'), 1), (0, 2, 3, 1))

    train_coord_data = np.load(f'{folder}/coord_train.npz')['coordinates']
    val_coord_data = np.load(f'{folder}/coord_validation.npz')['coordinates']

    train_image_data = np.load(f'{folder}/img_input_train_20.npz')['inputs']
    val_image_data = np.load(f'{folder}/img_input_validation_20.npz')['inputs']

    if version == 'v1':
        training_beam_output = beam_outputs(f'{folder}/beams_output_train.npz')
        validation_beam_output = beam_outputs(f'{folder}/beams_output_validation.npz')
    elif version == 'v2':
        training_beam_output = beam_outputs_v2(f'{folder}/beam_output_train.npz')
        validation_beam_output = beam_outputs_v2(f'{folder}/beams_output_validation.npz')
    else:
        raise Exception(f'Unknown version: {version}')

    coord, lidar = (train_coord_data, val_coord_data), (train_lidar_data, val_lidar_data)
    image = (train_image_data, val_image_data)
    coord_lidar = (train_coord_data, train_lidar_data), (val_coord_data, val_lidar_data)
    coord_lidar_image = (train_coord_data, train_lidar_data, train_image_data), \
                        (val_coord_data, val_lidar_data, val_image_data)

    possible_models = {
        'imperial': (imperial_model_fn, lidar),

        'beamsoup-coord': (beamsoup_coord_model_fn, coord),
        'beamsoup-lidar': (beamsoup_lidar_model_fn, lidar),
        'beamsoup-joint': (beamsoup_joint_model_fn, coord_lidar),

        'husky-coord': (nu_husky_coord_model_fn, coord),
        'husky-lidar': (nu_husky_lidar_model_fn, lidar),
        'husky-image': (nu_husky_image_model_fn, image),
        'husky-fusion': (nu_husky_fusion_model_fn, coord_lidar_image),

        # Baseline specifications provided by NU Husky are wrong and tensorflow throws an error.
        #   Unable to find the correct specifications
        'baseline-coord': (baseline_coord_model_fn, coord),
        'baseline-lidar': (baseline_lidar_model_fn, lidar),
        'baseline-image': (baseline_image_model_fn, image),
        'baseline-fusion': (baseline_fusion_model_fn, coord_lidar_image),

        'southampton': (southampton_model_fn, coord_lidar)
    }

    model_fn, (train_input, val_input) = possible_models[model_name]
    return model_fn, train_input, training_beam_output, val_input, validation_beam_output
