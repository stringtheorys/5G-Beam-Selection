
import tensorflow as tf


def _coord_model():
    _input_coord = tf.keras.layers.Input(shape=(2, 1))
    _coord_layer = tf.keras.layers.Dense(1024, activation='relu')(_input_coord)
    _coord_layer = tf.keras.layers.Dense(512, activation='relu')(_coord_layer)
    _coord_layer = tf.keras.layers.Dense(256, activation='relu')(_coord_layer)
    return _input_coord, _coord_layer


def _lidar_model():
    _input_layer = tf.keras.layers.Input(shape=(20, 200, 1))
    _lidar_layer = tf.keras.layers.Conv2D(30, kernel_size=(11, 11))(_input_layer)

    _lidar_layer = tf.keras.layers.Conv2D(25, kernel_size=(9, 9))(_lidar_layer)
    _lidar_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 1))(_lidar_layer)

    _lidar_layer = tf.keras.layers.Conv2D(20, kernel_size=(7, 7))(_lidar_layer)
    _lidar_layer = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(_lidar_layer)

    _lidar_layer = tf.keras.layers.Conv2D(15, kernel_size=(5, 5))(_lidar_layer)
    _lidar_layer = tf.keras.layers.Conv2D(10, kernel_size=(3, 3))(_lidar_layer)
    _lidar_layer = tf.keras.layers.Conv2D(1, kernel_size=(1, 1))(_lidar_layer)
    return _input_layer, _lidar_layer


def _image_model():
    _input_layer = tf.keras.layers.Input((9234, 48, 81))
    _image_layer = tf.keras.layers.Conv2D(8, (3, 3))

    for _ in range(3):
        _image_layer = tf.keras.layers.Conv2D(8, (2, 2))(_image_layer)
        _image_layer = tf.keras.layers.MaxPooling2D((2, 1))

    return _input_layer, _image_layer


def baseline_coord_model_fn():
    coord_input, coord_layer = _coord_model()
    output_layer = tf.keras.layers.Dense(256, activation='softmax')(coord_layer)
    return tf.keras.models.Model(inputs=coord_input, outputs=output_layer)


def baseline_lidar_model_fn():
    lidar_input, lidar_layer = _lidar_model()
    output_layer = tf.keras.layers.Dense(256, activation='softmax')(lidar_layer)
    return tf.keras.models.Model(inputs=lidar_input, outputs=output_layer)


def baseline_image_model_fn():
    image_input, image_layer = _image_model()
    output_layer = tf.keras.layers.Dense(256, activation='softmax')(image_layer)
    return tf.keras.models.Model(inputs=image_input, outputs=output_layer)


def baseline_fusion_model_fn():
    coord_input, coord_layer = _coord_model()
    lidar_input, lidar_layer = _lidar_model()
    image_input, image_layer = _image_model()
    concat_layer = tf.keras.layers.concatenate([coord_layer, lidar_layer, image_layer])

    layer = tf.keras.layers.Dense(512, activation='relu')(concat_layer)
    output_layer = tf.keras.layers.Dense(256, activation='softmax')(layer)
    return tf.keras.models.Model(inputs=[coord_input, image_input, lidar_input], outputs=output_layer)
