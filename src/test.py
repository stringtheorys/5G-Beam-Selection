
import matplotlib.pyplot as plt
import numpy as np

from core.dataset import beam_outputs, beams_log_scale


def testing_beam_output():
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    pos = 10

    raw_output = np.real(np.load('../data/beams_output_validation.npz')['output_classification'][pos].flatten())
    axs[0, 0].plot(np.arange(256), np.real(raw_output))
    axs[0, 0].set_title('Raw output')

    normalised_output = np.real(raw_output) / np.max(np.real(raw_output))
    axs[0, 1].plot(np.arange(256), normalised_output)
    axs[0, 1].set_title('Max Normalised output')

    axs[0, 2].plot(np.arange(256), normalised_output / np.sum(normalised_output))
    axs[0, 2].set_title('Max Sum Normalised output')

    axs[1, 0].plot(np.arange(256), raw_output / np.sum(raw_output))
    axs[1, 0].set_title('Sum normalised output')

    axs[1, 1].plot(np.arange(256), beams_log_scale(np.array([normalised_output]), 6)[0])
    axs[1, 1].set_title('Log scale of normalised')

    validation_output = beam_outputs('../data/beams_output_validation.npz')
    axs[1, 2].plot(np.arange(256), validation_output[pos])
    axs[1, 2].set_title('Imperial solution')

    plt.show()
