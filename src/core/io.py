import numpy as np
import tensorflow as tf
from typing import Tuple

from core.dataset import lidar_to_2d, beam_output
from models.beamsoup import beamsoup_coord_model, beamsoup_lidar_model, beamsoup_joint_model
from models.imperial import imperial_model
from models.nu_huskies import husky_coord_model, husky_lidar_model, husky_image_model, husky_fusion_model


def parse_model(model_name: str) -> Tuple[tf.keras.models.Sequential, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parsing the model name and training/validation data

    :param model_name: The model name
    :return: tensorflow model, training input and output, validation input and output
    """
    # Load the training and validation datasets
    train_lidar_data = np.transpose(np.expand_dims(lidar_to_2d('../data/lidar_train.npz'), 1), (0, 2, 3, 1))
    train_coord_data = np.load('../data/coord_train.npz')['coordinates']
    train_image_data = np.load('../data/img_input_train_20.npz')['inputs']
    training_beam_output = beam_output('../data/beams_output_train.npz')

    val_lidar_data = np.transpose(np.expand_dims(lidar_to_2d('../data/lidar_validation.npz'), 1), (0, 2, 3, 1))
    val_coord_data = np.load('../data/coord_validation.npz')['coordinates']
    val_image_data = np.load('../data/img_input_validation_20.npz')['inputs']
    validation_beam_output = beam_output('../data/beams_output_validation.npz')

    coord, lidar = (train_coord_data, val_coord_data), (train_lidar_data, val_lidar_data)
    image = (train_image_data, val_image_data)
    coord_lidar = (train_coord_data, train_lidar_data), (val_coord_data, val_lidar_data)
    coord_lidar_image = (train_coord_data, train_lidar_data, train_image_data), \
                        (val_coord_data, val_lidar_data, val_image_data)

    possible_models = {
        'imperial': (imperial_model, lidar),

        'beamsoup-coord': (beamsoup_coord_model, coord),
        'beamsoup-lidar': (beamsoup_lidar_model, lidar),
        'beamsoup-joint': (beamsoup_joint_model, coord_lidar),

        'husky-coord': (husky_coord_model, coord),
        'husky-lidar': (husky_lidar_model, lidar),
        'husky-image': (husky_image_model, image),
        'husky-fusion': (husky_fusion_model, coord_lidar_image),

        # Baseline specifications provided by NU Husky are wrong and tensorflow throws an error.
        #   Unable to find the correct specifications
        # 'baseline-coord': (baseline_coord_model, coord),
        # 'baseline-lidar': (baseline_lidar_model, lidar),
        # 'baseline-image': (baseline_image_model, image),
        # 'baseline-fusion': (baseline_fusion_model, coord_lidar_image)
    }

    model, (train_input, val_input) = possible_models[model_name]
    return model, train_input, training_beam_output, val_input, validation_beam_output
