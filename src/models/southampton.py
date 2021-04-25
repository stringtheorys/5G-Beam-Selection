

import tensorflow as tf


def _lidar_model(initialiser=tf.keras.initializers.HeUniform()):
    _input_layer = tf.keras.layers.Input(shape=(20, 200, 1))
    _lidar_layer = tf.keras.layers.Conv2D(5, 3, 1, padding='same', kernel_initializer=initialiser)(_input_layer)
    _lidar_layer = tf.keras.layers.BatchNormalization()(_lidar_layer)
    _lidar_layer = tf.keras.layers.PReLU(shared_axes=[1, 2])(_lidar_layer)

    for _ in range(2):
        _lidar_layer = tf.keras.layers.Conv2D(5, 3, 1, padding='same', kernel_initializer=initialiser)(_lidar_layer)
        _lidar_layer = tf.keras.layers.BatchNormalization()(_lidar_layer)
        _lidar_layer = tf.keras.layers.PReLU(shared_axes=[1, 2])(_lidar_layer)

        _lidar_layer = tf.keras.layers.Conv2D(5, 3, 2, padding='same', kernel_initializer=initialiser)(_lidar_layer)
        _lidar_layer = tf.keras.layers.BatchNormalization()(_lidar_layer)
        _lidar_layer = tf.keras.layers.PReLU(shared_axes=[1, 2])(_lidar_layer)

    _lidar_layer = tf.keras.layers.Conv2D(1, 3, (1, 2), padding='same', kernel_initializer=initialiser)(_lidar_layer)
    _lidar_layer = tf.keras.layers.BatchNormalization()(_lidar_layer)
    _lidar_layer = tf.keras.layers.PReLU(shared_axes=[1, 2])(_lidar_layer)

    _lidar_layer = tf.keras.layers.Flatten()(_lidar_layer)
    _lidar_layer = tf.keras.layers.Dense(512, activation='relu')(_lidar_layer)  # This was originally 16
    return _input_layer, _lidar_layer


def _coord_model():
    _input_layer = tf.keras.layers.Input(shape=(2, 1))

    _coord_layer = tf.keras.layers.Conv1D(20, 2, padding='same')(_input_layer)
    _coord_layer = tf.keras.layers.BatchNormalization()(_coord_layer)
    _coord_layer = tf.keras.layers.PReLU(shared_axes=[1, 2])(_coord_layer)

    _coord_layer = tf.keras.layers.Conv1D(10, 2, padding='same')(_coord_layer)
    _coord_layer = tf.keras.layers.BatchNormalization()(_coord_layer)
    _coord_layer = tf.keras.layers.PReLU(shared_axes=[1, 2])(_coord_layer)

    _coord_layer = tf.keras.layers.Flatten()(_coord_layer)
    _coord_layer = tf.keras.layers.Dense(512, activation='relu')(_coord_layer)
    return _input_layer, _coord_layer


def southampton_model_fn():
    _coord_input, _coord_layer = _coord_model()
    _lidar_input, _lidar_layer = _lidar_model()

    joint_layer = tf.keras.layers.concatenate([_lidar_layer, _coord_layer])
    joint_layer = tf.keras.layers.Dense(512, 'relu')(joint_layer)
    joint_layer = tf.keras.layers.Dense(256, 'softmax')(joint_layer)
    return tf.keras.models.Model(inputs=[_coord_input, _lidar_input], outputs=joint_layer)
