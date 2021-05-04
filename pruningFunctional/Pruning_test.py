import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.python.framework import tensor_shape
import numpy as np
import pruningToolkit
import os
import zipfile
import tempfile

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

def get_gzipped_model_size(model):
    _, new_pruned_keras_file = tempfile.mkstemp(".h5")
    print("Saving pruned model to: ", new_pruned_keras_file)
    tf.keras.models.save_model(model, new_pruned_keras_file, include_optimizer=False)
    
    # Zip the .h5 model file
    _, zip3 = tempfile.mkstemp(".zip")
    with zipfile.ZipFile(zip3, "w", compression=zipfile.ZIP_DEFLATED) as f:
        f.write(new_pruned_keras_file)
    print(
        "Size of ",str(model)," before compression: %.5f Mb"
        % (os.path.getsize(new_pruned_keras_file) / float(2 ** 20))
    )
    print(
        "Size of ",str(model)," after compression: %.5f Mb"
        % (os.path.getsize(zip3) / float(2 ** 20))
    )

def main():
    base_model = setup_model()
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
    
    pruned = pruningToolkit.pruneModel(base_model,0.8)    
    
    # print(base_model.get_config())
    # print('\n',"----------------",'\n')
    # print(pruned.get_config())
    
    
    #print('\n',"----------------",'\n')
    #print(pruned.weights)
    #print(pruningToolkit.currentSparcity(pruned.weights[0]))
    get_gzipped_model_size(pruned)
    # print('\n',"----------------",'\n')
    #print(pruningToolkit.currentSparcity(pruned))
    
    
    #print(weights[0])
    

if __name__ == "__main__":
    main()