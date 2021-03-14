import datetime as dt
import json

import numpy as np
import tensorflow as tf

# Lidar model
from core.common import get_beam_output, lidar_to_2d, model_top_metric_eval

lidar_layer = tf.keras.layers.Input()
lidar_layer = tf.math.divide(tf.add(tf.cast(lidar_layer, dtype=tf.float32), 2), 3)  # Scaling to [0,1] interval
lidar_layer = tf.keras.layers.GaussianNoise(0.005)(lidar_layer)  # 0.002
lidar_layer = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same')(lidar_layer)
lidar_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(lidar_layer)
lidar_layer = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same')(lidar_layer)
lidar_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(lidar_layer)
lidar_layer = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same')(lidar_layer)
lidar_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(lidar_layer)
lidar_layer = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(lidar_layer)
lidar_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(lidar_layer)
lidar_layer = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(lidar_layer)
lidar_layer = tf.keras.layers.Flatten()(lidar_layer)
lidar_layer = tf.keras.layers.Dense(400, activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(lidar_layer)

# Coord model
coord_layer = tf.keras.layers.Input()
coord_layer = tf.keras.layers.Dense(128, activation='relu')(coord_layer)
coord_layer = tf.keras.layers.GaussianNoise(0.002)(coord_layer)

# Combine the coord and lidar models
alignment_layer = tf.keras.layers.concatenate([lidar_layer, coord_layer])
alignment_layer = tf.keras.layers.Dense(600, activation='relu')(alignment_layer)
alignment_layer = tf.keras.layers.Dense(600, activation='relu')(alignment_layer)
alignment_layer = tf.keras.layers.Dense(500, activation='relu')(alignment_layer)
alignment_layer = tf.keras.layers.Dense(256, activation='softmax')(alignment_layer)

# Structure the model with both inputs
model = tf.keras.models.Sequential(inputs=[lidar_layer, alignment_layer], outputs=alignment_layer)

# Loss, optimiser and metrics for the model
loss = tf.keras.losses.CategoricalCrossentropy()
optimiser = tf.keras.optimizers.Adam()

top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top-1')
top10 = tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top-10')

# Tensorboard for logging of training info
log_directory = f'logs/beamsoup-{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}'
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
with open('beamsoup_agent_eval.json', 'w') as file:
    json.dumps({'correct': correct, 'top-k': top_k, 'throughput-ratio-k': throughput_ratio_k, 'history': history})
model.save('models/beamsoup-model')
