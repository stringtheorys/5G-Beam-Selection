from tensorflow.python.keras.metrics import MeanMetricWrapper
import tensorflow as tf
import numpy as np


def top_k_metrics(model, validation_input, validation_output):
    """
    Calculates the number of correct predictions, top k accuracy and throughput ratio for k=1,...,100

    :param model: Model for prediction
    :param validation_input: Validation input dataset
    :param validation_output: Validation output dataset
    :return: Tuple of the number of correct predictions and two lists for the top k accuracy and throughput ratio
        of length 100 representing k=1,...,100
    """
    predictions = np.argsort(model.predict(validation_input), axis=1)
    correct, top_k, throughput_ratio_k = 0, [], []
    best_throughput = np.sum(np.log2(np.max(validation_output, axis=1) + 1))
    for pos in range(100):
        correct += np.sum(predictions[:, -pos-1] == np.argmax(validation_output, axis=1))
        top_k.append(correct / validation_output.shape[0])
        throughput_ratio_k.append(np.sum(np.log2(np.max(np.take_along_axis(
            validation_output, predictions, axis=1)[:, -pos - 1:], axis=1) + 1)) / best_throughput)
    return top_k, throughput_ratio_k


class TopKThroughputRatio(MeanMetricWrapper):
    """
    Top K Throughput Ratio metric
    """

    def __init__(self, k, name):
        MeanMetricWrapper.__init__(self, self.throughput, name, k=k)

    @staticmethod
    def throughput(y_true, y_pred, k):
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

