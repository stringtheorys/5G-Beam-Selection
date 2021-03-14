"""
Testing file for training of centralised beam alignment agent
"""

import datetime as dt
import json

import numpy as np
import tensorflow as tf

from core.common import get_beam_output, lidar_to_2d, model_top_metric_eval

# Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(20, 200, 1)),
    tf.keras.layers.Conv2D(5, 3, 1, padding='same', kernel_initializer=tf.keras.initializers.HeUniform),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.PReLU(shared_axes=[1, 2]),
    tf.keras.layers.Conv2D(5, 3, 1, padding='same', kernel_initializer=tf.keras.initializers.HeUniform),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.PReLU(shared_axes=[1, 2]),
    tf.keras.layers.Conv2D(5, 3, 2, padding='same', kernel_initializer=tf.keras.initializers.HeUniform),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.PReLU(shared_axes=[1, 2]),
    tf.keras.layers.Conv2D(5, 3, 1, padding='same', kernel_initializer=tf.keras.initializers.HeUniform),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.PReLU(shared_axes=[1, 2]),
    tf.keras.layers.Conv2D(5, 3, 2, padding='same', kernel_initializer=tf.keras.initializers.HeUniform),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.PReLU(shared_axes=[1, 2]),
    tf.keras.layers.Conv2D(1, 3, (1, 2), padding='same', kernel_initializer=tf.keras.initializers.HeUniform),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.PReLU(shared_axes=[1, 2]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16),
    tf.keras.layers.ReLU(),
    # layers.Dropout(0.7),
    tf.keras.layers.Dense(256),
    tf.keras.layers.Softmax()
])

# Loss, optimiser and metrics for the model
loss = tf.keras.losses.CategoricalCrossentropy()
optimiser = tf.keras.optimizers.Adam()

top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top-1')
top10 = tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top-10')

# Tensorboard for logging of training info
log_directory = f'logs/centralised-{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_directory)

# Compile the model
model.compile(optimizer=optimiser, loss=loss, metrics=[top1, top10])

# Train the model
training_lidar_data = np.transpose(np.expand_dims(lidar_to_2d('data/lidar_train.npz'), 1), (0, 2, 3, 1))
training_beam_output, _ = get_beam_output('data/beams_output_train.npz')
validation_lidar_data = np.transpose(np.expand_dims(lidar_to_2d('data/lidar_validation.npz'), 1), (0, 2, 3, 1))
validation_beam_output, _ = get_beam_output('data/beams_output_validation.npz')
history = model.fit(x=training_lidar_data, y=training_beam_output, callbacks=tensorboard_callback, batch_size=16,
                    validation_data=(validation_lidar_data, validation_beam_output))

# Custom evaluation
correct, top_k, throughput_ratio_k = model_top_metric_eval(model, validation_lidar_data, validation_beam_output)
with open('centralised_agent_eval.json', 'w') as file:
    json.dumps({'correct': correct, 'top-k': top_k, 'throughput-ratio-k': throughput_ratio_k, 'history': history})
model.save('models/centralised-model')
