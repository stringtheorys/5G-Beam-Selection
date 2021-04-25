import numpy as np


def lidar_to_2d(lidar_filename: str):
    """
    Converts the lidar point cloud to a 2d representation
    (however why the implementation is this way is unknown)

    :param lidar_filename: Lidar dataset filename
    :return: 2D representation of the lidar dataset
    """
    lidar_data = np.load(lidar_filename)['input']
    lidar_zeros = np.zeros_like(lidar_data)[:, :, :, 1]

    lidar_zeros[np.max(lidar_data == 1, axis=-1)] = 1
    lidar_zeros[np.max(lidar_data == -2, axis=-1)] = -2
    lidar_zeros[np.max(lidar_data == -1, axis=-1)] = -1

    return lidar_zeros


def beams_log_scale(y, threshold_below_max):
    """
    This function makes no sense

    :param y: beams matrix
    :param threshold_below_max: threshold for values to be below
    :return: Updated beams
    """
    for i in range(0, y.shape[0]):
        output = y[i, :]
        output_log = 20 * np.log10(output + 1e-30)
        output[output_log < np.amax(output_log) - threshold_below_max] = 0
        y[i, :] = output / sum(output)
    return y


def beam_outputs(output_file: str, threshold: int = 6):
    """
    Generate the beam output based on the output classification file

    :param output_file: output file
    :param threshold: the beam threshold
    :return: Vector of beam output
    """
    y_matrix = np.abs(np.load(output_file)['output_classification'])
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


def beam_outputs_v2(beam_filename: str):
    """
    An updated beam output function that reshapes the output to a 256 vector then normalises the matrix

    :param beam_filename: the beam output filename
    :return: normalised beam output
    """
    beam_matrix = np.real(np.load(beam_filename)['output_classification']).reshape((-1, 256))
    return beam_matrix / beam_matrix.sum()


def output_dataset(folder: str = '../data', version: str = 'v1'):
    if version == 'v1':
        return (beam_outputs(f'{folder}/beams_output_train.npz'),
                beam_outputs(f'{folder}/beams_output_validation.npz'))
    elif version == 'v2':
        return (beam_outputs_v2(f'{folder}/beams_output_train.npz'),
                beam_outputs_v2(f'{folder}/beams_output_validation.npz'))
    else:
        raise Exception(f'Unknown version: {version}')


def coord_dataset(folder: str = '../data'):
    return (np.load(f'{folder}/coord_train.npz')['coordinates'],
            np.load(f'{folder}/coord_validation.npz')['coordinates'])


def lidar_dataset(folder: str = '../data'):
    return (np.transpose(np.expand_dims(lidar_to_2d(f'{folder}/lidar_train.npz'), 1), (0, 2, 3, 1)),
            np.transpose(np.expand_dims(lidar_to_2d(f'{folder}/lidar_validation.npz'), 1), (0, 2, 3, 1)))


def image_dataset(folder: str = '../data'):
    return (np.load(f'{folder}/img_input_train_20.npz')['inputs'],
            np.load(f'{folder}/img_input_validation_20.npz')['inputs'])


def coord_lidar_dataset(folder: str = '../data'):
    return [data for data in zip(coord_dataset(folder), lidar_dataset(folder))]


def coord_lidar_image_dataset(folder: str = '../data'):
    return [data for data in zip(coord_dataset(folder), lidar_dataset(folder), image_dataset(folder))]
