
import tensorflow as tf


def _lidar_model():
    _lidar_input = tf.keras.layers.Input(shape=(20, 200, 1))
    _lidar_layer = tf.math.divide(tf.add(tf.cast(_lidar_input, dtype=tf.float32), 2), 3)  # Scaling to [0,1] interval
    _lidar_layer = tf.keras.layers.GaussianNoise(0.005)(_lidar_layer)  # 0.002
    _lidar_layer = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(_lidar_layer)
    _lidar_layer = tf.keras.layers.MaxPooling2D((2, 2), (2, 2))(_lidar_layer)
    _lidar_layer = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(_lidar_layer)
    _lidar_layer = tf.keras.layers.MaxPooling2D((2, 2), (2, 2))(_lidar_layer)
    _lidar_layer = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(_lidar_layer)
    _lidar_layer = tf.keras.layers.MaxPooling2D((2, 2), (2, 2))(_lidar_layer)
    _lidar_layer = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(_lidar_layer)
    _lidar_layer = tf.keras.layers.MaxPooling2D((2, 2), (2, 2))(_lidar_layer)
    _lidar_layer = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(_lidar_layer)
    _lidar_layer = tf.keras.layers.Flatten()(_lidar_layer)
    _lidar_layer = tf.keras.layers.Dense(400, activation='relu', kernel_regularizer='l2')(_lidar_layer)
    return _lidar_input, _lidar_layer


def _coord_model():
    _coord_input = tf.keras.layers.Input(shape=(2,))
    _coord_layer = tf.keras.layers.Dense(128, activation='relu')(_coord_input)
    _coord_layer = tf.keras.layers.GaussianNoise(0.002)(_coord_layer)
    return _coord_input, _coord_layer


def beamsoup_lidar_model_fn():
    lidar_input, lidar_layer = _lidar_model()
    output_layer = tf.keras.layers.Dense(256, activation='softmax')(lidar_layer)
    return tf.keras.models.Model(inputs=lidar_input, outputs=output_layer)


def beamsoup_coord_model_fn():
    coord_input, coord_layer = _coord_model()
    output_layer = tf.keras.layers.Dense(256, activation='softmax')(coord_layer)
    return tf.keras.models.Model(inputs=coord_input, outputs=output_layer)


def beamsoup_joint_model_fn():
    lidar_input, lidar_layer = _lidar_model()
    coord_input, coord_layer = _coord_model()

    joint_layer = tf.keras.layers.concatenate([lidar_layer, coord_layer])
    joint_layer = tf.keras.layers.Dense(600, activation='relu')(joint_layer)
    joint_layer = tf.keras.layers.Dense(600, activation='relu')(joint_layer)
    joint_layer = tf.keras.layers.Dense(500, activation='relu')(joint_layer)
    joint_layer = tf.keras.layers.Dense(256, activation='softmax')(joint_layer)

    return tf.keras.models.Model(inputs=[coord_input, lidar_input], outputs=joint_layer)
