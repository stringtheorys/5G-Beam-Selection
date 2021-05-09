"""
Module containing importable pruning functionality
"""
import numpy as np
import os
import zipfile
import tempfile

import tensorflow as tf


def genSparceMask(modelWeights):
    
    
    def masker(elem):
        return int(bool(elem))
    
    myfunc = np.vectorize(masker)
    
    return myfunc(modelWeights)

def currentsparsity(modelWeights):
    
    sparceMask = genSparceMask(modelWeights)
    sparsity = 1 - (np.sum(sparceMask) / np.product(sparceMask.shape))
    
    return sparsity
    

def pruneLayer(modelWeights, desiredSparsity):
    
    shape = modelWeights.shape
    numElements = np.product(shape)
    modelWeights.flatten()

    curSpars = currentsparsity(modelWeights)
    
    if desiredSparsity <= curSpars:
        return modelWeights
    
    weights = np.abs(modelWeights)
    sortedWeights = np.sort(weights,kind='quicksort',axis=None)
    pruneThreshold = sortedWeights[int(desiredSparsity*numElements) - 1]    
    
    def mask(elem):
        return elem * float(np.abs(elem) > pruneThreshold)
    myfunc = np.vectorize(mask)

    return np.reshape(myfunc(modelWeights), shape)

def pruneModel(Model, desiredsparsity):
    
    def apply_pruning_to_dense(layer):
        if isinstance(layer, tf.keras.layers.Dense):
            units = layer.units
            activation = layer.activation
            prunedWeights = pruneLayer(layer.weights[0].numpy(),desiredsparsity)
            init = tf.constant_initializer(prunedWeights)
            
            
            bias_choises = ('zeros',tf.keras.initializers.constant(layer.bias.numpy()))
            bias_initializer = bias_choises[layer.use_bias]

            layer = tf.keras.layers.Dense(units = units,
                                          activation = activation,
                                          kernel_initializer = init,
                                          bias_initializer = bias_initializer
                                          )
            
        return layer
    
    prunedModel = tf.keras.models.clone_model(Model, clone_function=apply_pruning_to_dense)
    
    return prunedModel

def get_gzipped_model_size(model):
    _, new_pruned_keras_file = tempfile.mkstemp(".h5")
    print("Saving pruned model to: ", new_pruned_keras_file)
    tf.keras.models.save_model(model, new_pruned_keras_file, include_optimizer=False)
    
    # Zip the .h5 model file
    _, zip3 = tempfile.mkstemp(".zip")
    with zipfile.ZipFile(zip3, "w", compression=zipfile.ZIP_DEFLATED) as f:
        f.write(new_pruned_keras_file)
        
    before = os.path.getsize(new_pruned_keras_file) / float(2 ** 20)
    after = os.path.getsize(zip3) / float(2 ** 20)
    print(
        "Size of model before compression: %.5f Mb"
        % (before)
    )
    print(
        "Size of model after compression: %.5f Mb"
        % (after)
    )
    
    perDif = ( (before - after )/ before ) * 100
    
    return after, perDif

def get_gzipped_modelWeights_size(model):
    _, new_pruned_keras_file = tempfile.mkstemp(".h5")
    print("Saving pruned model weights to: ", new_pruned_keras_file)
    model.save_weights(new_pruned_keras_file)
    
    # Zip the .h5 model file
    _, zip3 = tempfile.mkstemp(".zip")
    with zipfile.ZipFile(zip3, "w", compression=zipfile.ZIP_DEFLATED) as f:
        f.write(new_pruned_keras_file)
        
    before = os.path.getsize(new_pruned_keras_file) / float(2 ** 20)
    after = os.path.getsize(zip3) / float(2 ** 20)
    print(
        "Size of model before compression: %.5f Mb"
        % (before)
    )
    print(
        "Size of model after compression: %.5f Mb"
        % (after)
    )
    
    perDif = ( (before - after )/ before ) * 100
    
    return after, perDif






