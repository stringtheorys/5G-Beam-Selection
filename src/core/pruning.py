"""
Module containing importable pruning functionality
"""

import tensorflow as tf


def generated_sparse_mask(model_weights):
    def masker(elem):
        return int(bool(elem))

    shape = model_weights.shape
    flat_mask = tf.map_fn(fn=masker, elems=tf.reshape(model_weights, [-1]), fn_output_signature=tf.int32)

    return tf.reshape(flat_mask, shape)


def weights_sparsity(model_weights):
    sparse_mask = generated_sparse_mask(model_weights)

    return 1 - (tf.reduce_sum(sparse_mask) / tf.reduce_prod(sparse_mask.shape))


def prune_layer(model_weights, desired_sparsity):
    if desired_sparsity <= weights_sparsity(model_weights):
        return model_weights

    shape = model_weights.shape
    sorted_weights = tf.sort(model_weights.numpy().flatten())
    prune_threshold = sorted_weights[int(desired_sparsity * tf.reduce_prod(shape)) - 1]

    def mask(elem):
        return elem * float(prune_threshold < elem)

    pruned_weights_flat = tf.map_fn(fn=mask, elems=tf.reshape(model_weights, [-1]), fn_output_signature=tf.float32)
    pruned_weights = tf.reshape(pruned_weights_flat, shape)

    return pruned_weights


def prune_model(model, desired_sparsity):
    def apply_pruning_to_dense(layer):
        if isinstance(layer, tf.keras.layers.Dense):
            pruned_weights = prune_layer(layer.weights[0], desired_sparsity)
            kernel_initializer = tf.keras.initializers.constant(pruned_weights)

            bias_initializer = tf.keras.initializers.constant(layer.bias.numpy()) if layer.use_bias else 'zeros'

            layer = tf.keras.layers.Dense(units=layer.units, activation=layer.activation,
                                          kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        return layer

    return tf.keras.models.clone_model(model, clone_function=apply_pruning_to_dense)
