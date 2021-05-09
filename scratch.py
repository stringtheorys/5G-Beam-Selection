import os
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from core.dataset import output_dataset
from core.metrics import TopKThroughputRatio
from core.pruningToolkit import pruneModel , get_gzipped_model_size, get_gzipped_modelWeights_size
from models import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

modelN = [['imperial','centralised-imperial/model'],
          ['beamsoup-joint','centralised-beamsoup-joint/model'],
          ['southampton','centralised-southampton/model'],
          #('husky-fusion','centralised-husky-fusion/model')
          ]

#%% PRUNING STEP - only pruning final model!
out = []
sp = np.arange(5,100,5)

for mod in modelN:
    
    print(mod[0])
    
    _, dataset_fn = models[mod[0]]

    metrics = [
        tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top-1-accuracy'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top-10-accuracy'),
        TopKThroughputRatio(k=1, name='top-1-throughput'),
        TopKThroughputRatio(k=10, name='top-10-throughput')
    ]


    model = tf.keras.models.load_model('../results/models/'+mod[1],
                                       custom_objects={'TopKThroughputRatio': TopKThroughputRatio})


    _, validation_input = dataset_fn()
    _, validation_output = output_dataset()

    base = model.evaluate(validation_input, validation_output)
    
    a = []
    for iter in sp:
        before = perf_counter()
        pruned_model = pruneModel(model,iter/100)
        size_model , perDif_model = get_gzipped_model_size(pruned_model)
        size_weights , perDif_weights = get_gzipped_modelWeights_size(pruned_model)
        pruned_model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=metrics)
        after = perf_counter()
        pruneTime = after - before
        data = pruned_model.evaluate(validation_input, validation_output)
        data.extend([pruneTime,size_model,size_weights,perDif_model,perDif_weights])
        a.append(data)
    aMod = np.reshape(a,(len(a),len(a[-1])))
    out.append(aMod)
    print(64*'-','\n')

#%% PLOT GRAPHS - saves to ../results/graphs
labels = ['Loss',
          'Top-1-Accuracy',
          'Top-10-Accuracy',
          'Top-1-Throughput',
          'Top-10-Throughput',
          'Total-Time-to-Prune-s',
          'Zipped-Model-Size',
          'Zipped-Model-Size-Weights-Only',
          'Percentage-Compression-Whole-Model',
          'Percentage-Compression-Weights-Only'
          ]

def getFirst(List):
    return [item[0] for item in List]

for iter in range(len(out[-1][-1])):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for test in range(len(out)):
        y = out[test][:,iter]
        ax.plot(sp,y)
        ax.set_xlabel('Percentage Sparsity / %',fontsize=14)
        ax.set_ylabel(labels[iter],fontsize=14)
        if iter == len(out[-1][-1])-1:
            pass
        else:
            ax.legend(getFirst(modelN))
    
    fig.savefig('../results/graphs/'+labels[iter]+'.png')













