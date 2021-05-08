import os

import tensorflow as tf

from core.dataset import output_dataset
from core.metrics import TopKThroughputRatio
from models import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

_, dataset_fn = models['imperial']

model = tf.keras.models.load_model('../results/models/centralised-imperial/model',
                                   custom_objects={'TopKThroughputRatio': TopKThroughputRatio})
_, validation_input = dataset_fn()
_, validation_output = output_dataset()

print(model.evaluate(validation_input, validation_output))
