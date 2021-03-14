"""
Testing file for training of federated learning beam alignment agent
"""

import collections
import json

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from core.common import lidar_to_2d, get_beam_output, model_top_metric_eval


def get_vehicle_dataset(total_vehicles, vehicle):
    lidar_data = np.transpose(np.expand_dims(lidar_to_2d('data/lidar_train.npz'), 1), (0, 2, 3, 1))
    beam_output, _ = get_beam_output('data/beams_output_train.npz')

    # Split lidar data and beam labels
    split_lidar_data = lidar_data[vehicle * int(lidar_data.shape[0] / total_vehicles):(vehicle + 1) * int(lidar_data.shape[0] / total_vehicles), :, :, :]
    split_beam_output = beam_output[vehicle * int(beam_output.shape[0] / total_vehicles):(vehicle + 1) * int(beam_output.shape[0] / total_vehicles), :]

    dataset_train = tf.data.Dataset.from_tensor_slices((list(split_lidar_data.astype(np.float32)),
                                                        list(split_beam_output.astype(np.float32))))
    return dataset_train


def get_validation_dataset():
    validation_lidar_data = np.transpose(np.expand_dims(lidar_to_2d('data/lidar_validation.npz'), 1), (0, 2, 3, 1))
    validation_beam_output, _ = get_beam_output('data/beams_output_validation.npz')
    dataset_test = tf.data.Dataset.from_tensor_slices((list(validation_lidar_data.astype(np.float32)),
                                                       list(validation_beam_output.astype(np.float32))))
    return dataset_test


def preprocess(dataset):
    def batch_format_fn(elem_1, elem_2):
        return collections.OrderedDict(x=elem_1, y=elem_2)
    return dataset.repeat(1).shuffle(20).batch(16).map(batch_format_fn).prefetch(10)


def model_fn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(20, 200, 1)),
        tf.keras.layers.Conv2D(5, 3, 1, padding='same'),  # kernel_initializer=initializers.HeUniform),
        tf.keras.layers.BatchNormalization(axis=3),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
        tf.keras.layers.Conv2D(5, 3, 1, padding='same'),  # kernel_initializer=initializers.HeUniform),
        tf.keras.layers.BatchNormalization(axis=3),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
        tf.keras.layers.Conv2D(5, 3, 2, padding='same'),  # kernel_initializer=initializers.HeUniform),
        tf.keras.layers.BatchNormalization(axis=3),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
        tf.keras.layers.Conv2D(5, 3, 1, padding='same'),  # kernel_initializer=initializers.HeUniform),
        tf.keras.layers.BatchNormalization(axis=3),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
        tf.keras.layers.Conv2D(5, 3, 2, padding='same'),  # , kernel_initializer=initializers.HeUniform),
        tf.keras.layers.BatchNormalization(axis=3),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
        tf.keras.layers.Conv2D(1, 3, (1, 2), padding='same'),  # , kernel_initializer=initializers.HeUniform),
        tf.keras.layers.BatchNormalization(axis=3),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16),
        tf.keras.layers.ReLU(),
        # layers.Dropout(0.7),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Softmax()
    ])
    top_1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top-1')
    top_10 = tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top-10')
    return tff.learning.from_keras_model(model,
                                         input_spec=preprocess(get_validation_dataset()).element_spec,
                                         loss=tf.keras.losses.CategoricalCrossentropy(),
                                         metrics=[top_1, top_10])


# Arguments
num_vehicles, aggregation_rounds = 2, 30

# initialise the federated learning
federated_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(lr=5e-3),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(lr=.25))
federated_evaluation = tff.learning.build_federated_evaluation(model_fn)
federated_state = federated_process.initialize()

# Generate vehicle training dataset and the testing dataset
vehicle_training_dataset = [preprocess(get_vehicle_dataset(num_vehicles, v)) for v in range(num_vehicles)]
validation_data = [preprocess(get_validation_dataset())]
custom_val_lidar = np.transpose(np.expand_dims(lidar_to_2d('data/lidar_validation.npz'), 1), (0, 2, 3, 1))
custom_val_beam, _ = get_beam_output('data/beams_output_validation.npz')

# Evaluation metrics
metrics = {'training-top-1': [], 'training-top-10': [], 'testing-top-1': [], 'testing-top-10': [],
           'correct': [], 'top-k': [], 'throughput_ratio': []}

# Federated Training
for round_num in range(aggregation_rounds):
    federated_state, training_metrics = federated_process.next(federated_state, vehicle_training_dataset)
    testing_metrics = federated_evaluation(federated_state.model, validation_data)

    # Save metrics
    metrics['training-top-1'].append(training_metrics['top-1'])
    metrics['training-top-10'].append(training_metrics['top-10'])
    metrics['loss'].append(training_metrics['loss'])
    metrics['testing-top-1'].append(testing_metrics['top-1'])
    metrics['testing-top-10'].append(testing_metrics['top-10'])

    # Custom evaluations
    correct, top_k, throughput_ratio = model_top_metric_eval(federated_state.model, custom_val_lidar, custom_val_beam)
    metrics['correct'].append(correct)
    metrics['top-k'].append(top_k)
    metrics['throughput-ratio-k'].append(throughput_ratio)

# Save the agent results
with open('federated-agent-metrics.json') as file:
    json.dump(metrics, file)
federated_state.model.save('models/federated-model')
