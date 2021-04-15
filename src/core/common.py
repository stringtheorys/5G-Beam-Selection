
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.metrics import MeanMetricWrapper

# from models.baseline import baseline_image_model, baseline_lidar_model, baseline_coord_model, baseline_fusion_model
from models.beamsoup import beamsoup_coord_model, beamsoup_lidar_model, beamsoup_joint_model
from models.imperial import imperial_model
from models.nu_huskies import husky_coord_model, husky_lidar_model, husky_image_model, husky_fusion_model


class TopKThroughputRatio(MeanMetricWrapper):
    """
    Top K Throughput Ratio metric
    """

    def __init__(self, k, name):
        MeanMetricWrapper.__init__(self, self.throughput, name, k=k)

    def throughput(self, y_true, y_pred, k):
        """
        Finds the throughput ratio of the top k beams compared to the optimal beam
            Algorithm works by
            1. Sorting the y pred in reverse order and collecting the first k responses indices (top k responses)
            2. Gather the value from the y true array using the indices from the top k y pred
            3. Reduce the maximum y true value for each beam
            4. Get the ratio between the max y true value and the max y true based on the predicted values

        :param y_true: True beam output
        :param y_pred: Predicted beam output
        :param k: top k beam
        :return: top k beam throughput ratio
        """
        return tf.divide(tf.reduce_max(tf.gather(y_true, tf.argsort(y_pred, direction='DESCENDING')[:, :k],
                                                 axis=1, batch_dims=1), axis=1), tf.reduce_max(y_true, axis=1))


def lidar_to_2d(lidar_data_path):
    lidar_data = np.load(lidar_data_path)['input']
    lidar_zeros = np.zeros_like(lidar_data)[:, :, :, 1]

    lidar_zeros[np.max(lidar_data == 1, axis=-1)] = 1
    lidar_zeros[np.max(lidar_data == -2, axis=-1)] = -2
    lidar_zeros[np.max(lidar_data == -1, axis=-1)] = -1

    return lidar_zeros


def beams_log_scale(y, threshold_below_max):
    y_shape = y.shape
    for i in range(0, y_shape[0]):
        output = y[i, :]
        output_log = 20 * np.log10(output + 1e-30)
        output[output_log < np.amax(output_log) - threshold_below_max] = 0
        y[i, :] = output / sum(output)

    return y


def get_beam_output(output_file, threshold=6):
    output_cache_file = np.load(output_file)
    y_matrix = np.abs(output_cache_file['output_classification'])
    y_matrix /= np.max(y_matrix)
    num_classes = y_matrix.shape[1] * y_matrix.shape[2]

    # new ordering of the beams, provided by the Organizers
    y = np.zeros((y_matrix.shape[0], num_classes))
    for i in range(0, y_matrix.shape[0], 1):  # go over all examples
        codebook = np.absolute(y_matrix[i, :])  # read matrix
        rx_size = codebook.shape[0]  # 8 antenna elements
        for tx in range(codebook.shape[1]):  # 32 antenna elements
            for rx in range(rx_size):  # inner loop goes over receiver
                y[i, tx * rx_size + rx] = codebook[rx, tx]  # impose ordering

    return beams_log_scale(y, threshold)


def model_top_metric_eval(model, validation_lidar_data, validation_beam_output):
    predictions = np.argsort(model.predict(validation_lidar_data), axis=1)
    correct, top_k, throughput_ratio_k = 0, [], []
    best_throughput = np.sum(np.log2(np.max(validation_beam_output, axis=1) + 1))
    for pos in range(100):
        correct += np.sum(predictions[:, -1-pos] == np.argmax(validation_beam_output, axis=1))
        top_k.append(correct / validation_beam_output.shape[0])
        throughput_ratio_k.append(np.sum(np.log2(np.max(np.take_along_axis(
            validation_beam_output, predictions, axis=1)[:, -1-pos:], axis=1) + 1)) / best_throughput)
    return correct, top_k, throughput_ratio_k


def parse_model(model_name: str):
    """
    Parsing the model name and training/validation data

    :param model_name: The model name
    :return: tensorflow model, training input and output, validation input and output
    """
    # Load the training and validation datasets
    train_lidar_data = np.transpose(np.expand_dims(lidar_to_2d('../data/lidar_train.npz'), 1), (0, 2, 3, 1))
    train_coord_data = np.load('../data/coord_train.npz')['coordinates']
    train_image_data = np.load('../data/img_input_train_20.npz')['inputs']
    training_beam_output = get_beam_output('../data/beams_output_train.npz')

    val_lidar_data = np.transpose(np.expand_dims(lidar_to_2d('../data/lidar_validation.npz'), 1), (0, 2, 3, 1))
    val_coord_data = np.load('../data/coord_validation.npz')['coordinates']
    val_image_data = np.load('../data/img_input_validation_20.npz')['inputs']
    validation_beam_output = get_beam_output('../data/beams_output_validation.npz')

    coord, lidar, image = (train_coord_data, val_coord_data), (train_lidar_data, val_lidar_data), \
                          (train_image_data, val_image_data)
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

        # 'baseline-coord': (baseline_coord_model, coord),
        # 'baseline-lidar': (baseline_lidar_model, lidar),
        # 'baseline-image': (baseline_image_model, image),
        # 'baseline-fusion': (baseline_fusion_model, coord_lidar_image)
    }

    model, (train_input, val_input) = possible_models[model_name]
    return model, train_input, training_beam_output, val_input, validation_beam_output
