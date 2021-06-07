import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.python.framework import tensor_shape
import numpy as np
from core.pruningToolkit import pruneModel , get_gzipped_model_size, get_gzipped_modelWeights_size
from core.sparsityToolkit import to_sparse, to_dense
import os
import zipfile
import tempfile
import scipy.sparse as sp
import sys
import matplotlib.pyplot as plt

input_shape = [20]

def setup_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(20, input_shape=input_shape),
      tf.keras.layers.Flatten()
  ])
  return model

def setup_pretrained_weights():
  model = setup_model()

  model.compile(
      loss=tf.keras.losses.categorical_crossentropy,
      optimizer='adam',
      metrics=['accuracy']
  )

def main():
    base_model = setup_model()
    #bw = base_model.weights[0].numpy
    #model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model)
        
    #base_model.summary()
    #print(base_model.weights)
    # print(np.sum(base_model.weights[1]))
    # model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model)
    # model_for_pruning.summary()
    # print(np.sum(model_for_pruning.weights[2].numpy()))
    # print(pruningToolkit.currentSparcity(model_for_pruning.weights))
    
    # #tfmot.sparsity.keras.UpdatePruningStep()
    # model_for_pruning.compile(
    #   loss=tf.keras.losses.categorical_crossentropy,
    #   optimizer='adam',
    #   metrics=['accuracy']
    #   )
    

    # model_for_pruning.optimizer = tf.keras.optimizers.Adam()
    # step_callback = tfmot.sparsity.keras.UpdatePruningStep()
    # step_callback.set_model(model_for_pruning)
    # step_callback.on_train_begin()
    # step_callback.on_epoch_end(batch=-1)
    # print(model_for_pruning.weights)
    
    
    #print(pruningToolkit.genSparceMask(weights))
    #get_gzipped_model_size(base_model)
    
    #get_gzipped_modelWeights_size(base_model)
    
    #pruned = pruneModel(base_model,0.8)
    
    # print(base_model.get_config())
    # print('\n',"----------------",'\n')
    # print(pruned.get_config())
    
    
    #print('\n',"----------------",'\n')
    
    
    #w = pruned.weights[0].numpy()
    #wA = pruned.weights[0].numpy
    # global sp_w
    # sp_w = to_sparse(w,np.float32)
    # rev = to_dense(sp_w,np.float32)
    
    #print(w, sp_w)
    
    # _, temp = tempfile.mkstemp(".npz")
    # np.savez_compressed(temp, sp_w)
    # print(os.path.getsize(temp))
    
    
    # _, temp2 = tempfile.mkstemp(".npz")
    # np.savez_compressed(temp2, wA)
    # print(os.path.getsize(temp2))
    
    # _, temp5 = tempfile.mkstemp(".npz")
    # np.savez_compressed(temp5, w)
    # print(os.path.getsize(temp5))
    
    
    # _, new_pruned_keras_file = tempfile.mkstemp(".h5")
    # pruned.save_weights(new_pruned_keras_file)
    # # Zip the .h5 model file
    # _, zip32 = tempfile.mkstemp(".zip")
    # with zipfile.ZipFile(zip32, "w", compression=zipfile.ZIP_DEFLATED) as f:
    #     f.write(new_pruned_keras_file)
    # print(zip32, os.path.getsize(zip32))
    
    
    
    _, new_pruned_keras_file = tempfile.mkstemp(".h5")
    base_model.save_weights(new_pruned_keras_file)
    _, zip32 = tempfile.mkstemp(".zip")
    with zipfile.ZipFile(zip32, "w", compression=zipfile.ZIP_DEFLATED) as f:
        f.write(new_pruned_keras_file)

    baseSize = os.path.getsize(zip32)
    
    #----------------------------------------------
    
    sparsity = np.arange(0,105,5)
    sparsity = [val/100 for val in sparsity]
    weightsOnly = list()
    sparseMatrixSmallest = list()
    sparseMatrix = list()
    
    for spar in sparsity:
        pruned = pruneModel(base_model,spar)
        w = pruned.weights[0].numpy()
        sp_w_min = to_sparse(w,np.float16)
        sp_w = to_sparse(w,np.dtype(('U',1)))
        
        _ , temp = tempfile.mkstemp(".npz")
        np.savez_compressed(temp, w)
        weightsOnly.append(os.path.getsize(temp))
    
        _ , temp = tempfile.mkstemp(".npz")
        np.savez_compressed(temp, sp_w_min)
        sparseMatrixSmallest.append(os.path.getsize(temp))
        
        _ , temp = tempfile.mkstemp(".npz")
        np.savez_compressed(temp, sp_w)
        sparseMatrix.append(os.path.getsize(temp))
        
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sparsity,weightsOnly)
    ax.plot(sparsity,sparseMatrixSmallest)
    ax.plot(sparsity,sparseMatrix)
    ax.hlines(baseSize,-10,10)
    ax.vlines(0.6666,-10,3500)
    ax.set_xlim(0,1)
    ax.set_ylim(0,3000)
    ax.legend(['Weights Only','float16 Sparse','float32 Sparse'])
    
    #print(pruningToolkit.currentSparcity(pruned.weights[0]))
    #get_gzipped_modelWeights_size(pruned)
    # print('\n',"----------------",'\n')
    #print(pruningToolkit.currentSparcity(pruned))
    
    
    #print(weights[0])
    

if __name__ == "__main__":
    main()