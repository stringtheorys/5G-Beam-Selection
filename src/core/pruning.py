"""
Module containing importable pruning functionality
"""
import os
import tempfile
import zipfile

import numpy as np
import tensorflow as tf


def generate_sparsity_mask(model_weights):
    def masker(elem):
        return int(bool(elem))

    return np.vectorize(masker)(model_weights)


def current_sparsity(model_weights):
    sparse_mask = generate_sparsity_mask(model_weights)
    sparsity = 1 - (np.sum(sparse_mask) / np.product(sparse_mask.shape))

    return sparsity


def prune_layer(model_weights, desired_sparsity):
    if desired_sparsity <= current_sparsity(model_weights):
        return model_weights

    shape = model_weights.shape
    num_elements = np.product(shape)
    sorted_weights = np.sort(np.abs(model_weights), kind='quicksort', axis=None)
    prune_threshold = sorted_weights[int(desired_sparsity * num_elements) - 1]

    def mask(elem):
        return elem * float(np.abs(elem) > prune_threshold)

    return np.reshape(np.vectorize(mask)(model_weights), shape)


def prune_model(model, desired_sparsity, clone_model=True):
    def apply_pruning_to_dense(layer):
        if isinstance(layer, tf.keras.layers.Dense):
            init = tf.constant_initializer(prune_layer(layer.weights[0].numpy(), desired_sparsity))
            bias_initializer = tf.keras.initializers.constant(layer.bias.numpy()) if layer.use_bias else 'zeros'
            layer = tf.keras.layers.Dense(units=layer.units, activation=layer.activation, kernel_initializer=init,
                                          bias_initializer=bias_initializer)
        return layer

    if clone_model:
        return tf.keras.models.clone_model(model, clone_function=apply_pruning_to_dense)
    else:
        return [apply_pruning_to_dense(layer) for layer in model.layers]


def to_sparse(weights, dtype=np.float32):
    indexed = list(np.ndenumerate(weights))
    sparse_matrix = []

    def is_zero(elem):
        ind, val = elem
        if val != 0:
            sparse_matrix.append(np.array([float(ind[0]), float(ind[1]), val]))

    sparsify = np.vectorize(is_zero, signature='(0)->()')
    sparsify(indexed)

    return np.array(sparse_matrix, dtype)


def to_dense(sparse_matrix, dtype=np.float32):
    x = int(np.max(sparse_matrix[:, 1])) + 1
    y = int(sparse_matrix[-1][0]) + 1

    base = np.zeros((x, y), dtype)

    def setter(elem):
        x_ind, y_ind, val = elem
        base[int(x_ind)][int(y_ind)] = val

    array_setter = np.vectorize(setter, signature='(0)->()')
    array_setter(sparse_matrix)

    return base


def get_gzipped_model_size(model):
    _, new_pruned_keras_file = tempfile.mkstemp(".h5")
    tf.keras.models.save_model(model, new_pruned_keras_file, include_optimizer=False)

    _, zip_file = tempfile.mkstemp(".zip")
    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as file:
        file.write(new_pruned_keras_file)

    model_size = os.path.getsize(zip_file) / float(2 ** 20)
    return model_size


def get_gzipped_model_weights_size(model):
    _, new_pruned_keras_file = tempfile.mkstemp(".h5")
    model.save_weights(new_pruned_keras_file)

    _, zip_file = tempfile.mkstemp(".zip")
    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as file:
        file.write(new_pruned_keras_file)

    model_size = os.path.getsize(zip_file) / float(2 ** 20)
    return model_size
