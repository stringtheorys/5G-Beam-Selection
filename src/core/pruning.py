"""
Module containing importable pruning functionality
"""

import numpy as np
import tensorflow as tf


def generate_sparsity_mask(model_weights):
    def masker(elem):
        return int(bool(elem))

    shape = model_weights.shape
    mask_flat = tf.map_fn(fn=masker, elems=tf.reshape(model_weights, [-1]), fn_output_signature=tf.int32)

    return tf.reshape(mask_flat, shape)


def current_sparsity(model_weights):
    sparsity_mask = generate_sparsity_mask(model_weights)
    sparsity = 1 - (np.sum(sparsity_mask) / np.product(sparsity_mask.shape))

    return sparsity


def prune_layer(model_weights, desired_sparsity):
    if desired_sparsity <= current_sparsity(model_weights):
        return model_weights

    shape = model_weights.shape
    sorted_weights = np.sort(model_weights.numpy(), kind='quicksort', axis=None)
    prune_threshold = sorted_weights[int(desired_sparsity * np.product(model_weights.shape)) - 1]

    def mask(elem):
        return elem * float(elem > prune_threshold)

    pruned_weights_flat = tf.map_fn(fn=mask, elems=tf.reshape(model_weights, [-1]), fn_output_signature=tf.float32)
    return tf.reshape(pruned_weights_flat, shape)


def prune_model(model, desired_sparsity):
    def apply_pruning_to_dense(layer):
        if isinstance(layer, tf.keras.layers.Dense):
            pruned_weights = prune_layer(layer.weights[0], desired_sparsity)
            kernel_initialiser = tf.keras.initializers.constant(pruned_weights)
            bias_initialiser = tf.keras.initializers.constant(layer.bias.numpy()) if layer.use_bias else 'zeros'

            layer = tf.keras.layers.Dense(units=layer.units, activation=layer.activation,
                                          kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser)
        return layer

    return tf.keras.models.clone_model(model, clone_function=apply_pruning_to_dense)
