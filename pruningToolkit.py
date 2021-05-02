"""
Module containing importable pruning functionality
"""
import numpy as np
import tensorflow as tf

def genSparceMask(modelWeights):
    
    def masker(elem):
        return int(bool(elem))
    
    shape = modelWeights.shape
    maskFlat = tf.map_fn(fn = masker,
                     elems = tf.reshape(modelWeights,[-1]),
                     fn_output_signature = tf.int32
                     )
    
    mask = tf.reshape(maskFlat,shape)
    
    return mask

def currentSparcity(modelWeights):
    
    sparceMask = genSparceMask(modelWeights)
    sparcity = 1 - (np.sum(sparceMask) / np.product(sparceMask.shape))
    
    return sparcity
    

def pruneLayer(modelWeights, desiredSparcity):
    
    curSpars = currentSparcity(modelWeights)
    
    if desiredSparcity <= curSpars:
        return modelWeights
    
    shape = modelWeights.shape
    numElements = np.product(shape)
    
    weights = modelWeights.numpy()
    sortedWeights = np.sort(weights,kind='quicksort',axis=None)
    
    pruneThreshold = sortedWeights[int(desiredSparcity*numElements) - 1]

    def mask(elem):
        return elem * float(elem > pruneThreshold)
    
    prunedWeightsFlat = tf.map_fn(fn = mask ,
                            elems = tf.reshape(modelWeights,[-1]),
                            fn_output_signature = tf.float32
                            )
        
    prunedWeights = tf.reshape(prunedWeightsFlat, shape)
    
    return prunedWeights

def pruneModel(Model, desiredSparcity):
    
    def apply_pruning_to_dense(layer):
        if isinstance(layer, tf.keras.layers.Dense):
            
            units = layer.units
            activation = layer.activation
            prunedWeights = pruneLayer(layer.weights[0],desiredSparcity)
            kernel_initializer = tf.keras.initializers.constant(prunedWeights)
            
            if layer.use_bias:
                biases = layer.bias.numpy()
                bias_initializer = tf.keras.initializers.constant(biases)
            else:
                bias_initializer = 'zeros'

            layer = tf.keras.layers.Dense(units = units,
                                          activation = activation,
                                          kernel_initializer = kernel_initializer,
                                          bias_initializer = bias_initializer
                                          )
            
        return layer
    
    prunedModel = tf.keras.models.clone_model(Model, clone_function=apply_pruning_to_dense)
    
    return prunedModel

def compressModel():
    pass

def decompressModel():
    pass
    