"""
Testing file for training of centralised beam alignment agent
"""

import datetime as dt
import json

import numpy as np
import tensorflow as tf

from common import lidar_to_2d, get_beam_output, get_beam_output_no_normalization

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

top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top 1')
top10 = tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top 10')

# Tensorboard for logging of training info
log_directory = f'logs/{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_directory)

# Compile the model
model.compile(optimizer=optimiser, loss=loss, metrics=[top1, top10])

# Train the model
training_lidar_data = np.transpose(np.expand_dims(lidar_to_2d('data/lidar_train.npz'), 1), (0, 2, 3, 1))
training_beam_output, _ = get_beam_output('data/beams_output_train.npz')
model.fit(training_lidar_data, training_beam_output, callbacks=tensorboard_callback, batch_size=16)

# Evaluate the model
validation_lidar_data = np.transpose(np.expand_dims(lidar_to_2d('data/lidar_validation.npz'), 1), (0, 2, 3, 1))
validation_beam_output, _ = get_beam_output('data/beams_output_validation.npz')
model.evaluate(validation_lidar_data, validation_beam_output)

# Custom evaluation
beam_output_true, _ = get_beam_output_no_normalization('data/beams_output_validation.npz')
predictions = np.argsort(model.predict(validation_lidar_data, batch_size=100), axis=1)
top_k, throughput_ratio_k, correct = np.zeros(100), np.zeros(100), 0
for pos in range(100):
    correct += np.sum(predictions[:, -1-pos] == np.argmax(validation_beam_output, axis=1))
    top_k[pos] = correct / validation_beam_output.shape[0]
    throughput_ratio_k[pos] = np.sum(np.log2(np.max(np.take_along_axis(beam_output_true, predictions, axis=1)[:, -1-pos:], axis=1) + 1)) / np.sum(np.log2(np.max(beam_output_true, axis=1) + 1))

with open('centralised_agent_eval.json', 'w') as file:
    json.dumps([top_k, throughput_ratio_k])

# Save the model
model.save_weights('centralised_model')
