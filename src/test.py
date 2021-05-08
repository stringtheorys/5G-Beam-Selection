import os
import tempfile
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from core.dataset import beam_outputs, beams_log_scale, output_dataset
from core.pruning import prune_model
from models import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def test_beam_output():
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    pos = 0  # Vary this value to investigate different beam outputs.

    # Here we are testing the validation dataset as it is smaller
    # Load the data, flatten the results to a vector and the real component as imaginary it all zero
    true_output = np.real(np.load('../data/beams_output_validation.npz')['output_classification'][pos].flatten())

    # Plot the true output
    axs[0, 0].plot(np.arange(256), true_output)
    axs[0, 0].set_title('True output')

    # Plot the normalised true output using the max value or by the sum of values
    axs[0, 1].plot(np.arange(256), true_output / np.max(true_output))
    axs[0, 1].set_title('Max Normalised output')
    axs[0, 2].plot(np.arange(256), true_output / np.sum(true_output))
    axs[0, 2].set_title('Sum normalised output')

    # This is what the beam outputs does when it is flattening the true output however this seems to produce the
    #   output as the sum normalised output results
    max_normalised_output = true_output / np.max(true_output)
    axs[1, 0].plot(np.arange(256), max_normalised_output / np.sum(max_normalised_output))
    axs[1, 0].set_title('Max Sum Normalised output')

    # Using the flattened with the beam log scaling function
    axs[1, 1].plot(np.arange(256), beams_log_scale(np.array([true_output / np.sum(true_output)]), 6)[pos])
    axs[1, 1].set_title('Log scale of normalised')

    # Compare these beam outputs to the imperial get beam outputs function
    validation_output = beam_outputs('../data/beams_output_validation.npz')
    axs[1, 2].plot(np.arange(256), validation_output[pos])
    axs[1, 2].set_title('Imperial solution')
    plt.show()


def test_models():
    for name, (model_fn, dataset_fn) in models.items():
        print(name)
        try:
            model = model_fn()
            _, validation_input = dataset_fn()
            model(validation_input)
        except Exception as e:
            print(e)


def test_beam_output_v2():
    model_fn, dataset_fn = models['imperial']
    model = tf.keras.models.load_model('../results/models/centralised-imperial-v2/model')
    _, validation_input = dataset_fn()
    _, validation_output = output_dataset(version='v2')

    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    for ax in axs.flatten():
        pos = np.random.randint(0, len(validation_input) - 1)
        ax.set_title(f'Pos: {pos}')
        ax.plot(np.arange(256), validation_output[pos], label='True')
        ax.plot(np.arange(256), model(np.array([validation_input[pos]]))[0], label='Predicted')
    axs[0, 0].legend()
    plt.show()


def gzip_model(test_model):
    _, new_pruned_keras_file = tempfile.mkstemp(".h5")
    print("Saving pruned model to: ", new_pruned_keras_file)
    tf.keras.models.save_model(test_model, new_pruned_keras_file, include_optimizer=False)

    # Zip the .h5 model file
    _, zip3 = tempfile.mkstemp(".zip")
    with zipfile.ZipFile(zip3, "w", compression=zipfile.ZIP_DEFLATED) as f:
        f.write(new_pruned_keras_file)
    print(f"Size of model before compression: {os.path.getsize(new_pruned_keras_file) / float(2 ** 20):.5f} Mb")
    print(f"Size of model after compression: {os.path.getsize(zip3) / float(2 ** 20):.5f} Mb")


def test_pruning(input_shape=(20,)):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, input_shape=input_shape),
        tf.keras.layers.Flatten()
    ])

    pruned = prune_model(model, 0.8)
    gzip_model(pruned)


if __name__ == '__main__':
    test_beam_output()
    # test_beam_output_v2()
