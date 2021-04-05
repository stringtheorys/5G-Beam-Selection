
import numpy as np
import tensorflow as tf

from core.common import TopKThroughputRatio
from core.models import beamsoup_lidar_layers, beamsoup_coord_layers


def southampton_training():
    # Loss and optimisers
    loss = tf.keras.losses.MeanSquaredError()
    optimiser = tf.keras.optimizers.Adam()

    # Metrics
    top1_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top-1-accuracy')
    top10_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top-10-accuracy')
    top1_throughput = TopKThroughputRatio(k=1, name='top-1-throughput')
    top10_throughput = TopKThroughputRatio(k=10, name='top-10-throughput')

    # Beamsoup model
    lidar_input, lidar_layer = beamsoup_lidar_layers()
    coord_input, coord_layer = beamsoup_coord_layers()
    alignment_layer = tf.keras.layers.concatenate([lidar_layer, coord_layer])
    alignment_layer = tf.keras.layers.Dense(600, activation='relu')(alignment_layer)
    alignment_layer = tf.keras.layers.Dense(600, activation='relu')(alignment_layer)
    alignment_layer = tf.keras.layers.Dense(500, activation='relu')(alignment_layer)
    alignment_layer = tf.keras.layers.Dense(256, activation='softmax')(alignment_layer)
    model = tf.keras.models.Model(inputs=[lidar_input, coord_input], outputs=alignment_layer)

    # Compile the model
    model.compile(optimizer=optimiser, loss=loss,
                  metrics=[top1_accuracy, top10_accuracy, top1_throughput, top10_throughput])

    # Load the training and validation data
    training_input = [np.load('../data/lidar_train.npz')['input'], np.load('../data/coord_train.npz')['coordinates']]
    validation_input = [np.load('../data/lidar_validation.npz')['input'], np.load('../data/coord_validation.npz')['coordinates']]
    training_output = np.load('../data/beams_output_train.npz')['output_classification'].reshape(9234, 256)
    validation_output = np.load('../data/beams_output_validation.npz')['output_classification'].reshape(1960, 256)

    # Train the model
    history = model.fit(x=training_input, y=training_output, batch_size=16, epochs=10,
                        validation_data=(validation_input, validation_output))
    # Save the model
    tf.keras.models.save_model(model, '../models/southampton-model/model')
