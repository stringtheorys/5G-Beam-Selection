

import matplotlib.pyplot as plt
import json
import numpy as np


def plt_evaluation(filename):
    with open(filename) as file:
        data = json.load(file)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(100), data['top-k'])
    ax.set_title('Top K Accuracy')
    plt.savefig('top-k-accuracy.png')
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(100), data['throughput-ratio-k'])
    ax.set_title('Top K Throughput Ratio')
    plt.savefig('top-k-throughput-ratio.png')
    plt.show()

    initial_processing = 100
    fig, ax = plt.subplots(figsize=(8, 5))
    for numerology, beam_time in [(0, 1), (1, 0.5), (2, 0.25), (3, 0.165)]:
        time_taken = initial_processing + np.arange(beam_time, beam_time * 101, beam_time)
        ax.plot(np.arange(100), np.array(data['throughput-ratio-k']) / time_taken,
                label=f'Numerology: {numerology}')
    ax.set_title('Top K Throughput Ratio over beam search time required')
    ax.set_ylabel('Ratio between the throughput ratio and the search time')
    ax.set_xlabel('Top K Beams')
    plt.legend()
    plt.savefig('top-k-throughput-ratio-over-time.png')
    plt.show()


plt_evaluation('../agents/centralised_agent_eval.json')
