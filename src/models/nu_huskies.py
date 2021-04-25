
import tensorflow as tf


def _coord_model():
    _input_coord = tf.keras.layers.Input(shape=(2, 1))
    _coord_layer = tf.keras.layers.Conv1D(20, 2, padding='same', activation='relu')(_input_coord)
    _coord_layer = tf.keras.layers.Conv1D(10, 2, padding='same', activation='relu')(_coord_layer)
    _coord_layer = tf.keras.layers.MaxPooling1D(padding='same')(_coord_layer)

    _coord_layer = tf.keras.layers.Conv1D(20, 2, padding='same', activation='relu')(_coord_layer)
    _coord_layer = tf.keras.layers.Conv1D(10, 2, padding='same', activation='relu')(_coord_layer)
    _coord_layer = tf.keras.layers.MaxPooling1D(padding='same')(_coord_layer)

    _coord_layer = tf.keras.layers.Flatten()(_coord_layer)
    _coord_layer = tf.keras.layers.Dense(1024, activation='relu')(_coord_layer)
    _coord_layer = tf.keras.layers.Dense(512, activation='relu')(_coord_layer)

    return _input_coord, _coord_layer


def _convolutional_module(input_block, channel, drop_prob, dropout=True):
    _conv_layer = tf.keras.layers.Conv2D(channel, (3, 3), padding='same', activation='relu')(input_block)
    _conv_layer = tf.keras.layers.Conv2D(channel, (3, 3), padding='same', activation='relu')(_conv_layer)
    _conv_layer = tf.keras.layers.Add()([_conv_layer, input_block])
    if dropout:
        _conv_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(_conv_layer)
        _conv_layer = tf.keras.layers.Dropout(drop_prob)(_conv_layer)
    return _conv_layer


def _lidar_model(input_shape=(20, 200, 1), channel=32, drop_prob=0.2):
    _input_layer = tf.keras.layers.Input(shape=input_shape)
    _lidar_layer = tf.keras.layers.Conv2D(channel, (3, 3), activation='relu', padding='same')(_input_layer)

    _lidar_layer = _convolutional_module(_lidar_layer, channel, drop_prob)
    _lidar_layer = _convolutional_module(_lidar_layer, channel, drop_prob)
    _lidar_layer = _convolutional_module(_lidar_layer, channel, drop_prob)
    _lidar_layer = _convolutional_module(_lidar_layer, channel, drop_prob, False)

    _lidar_layer = tf.keras.layers.Flatten()(_lidar_layer)
    _lidar_layer = tf.keras.layers.Dense(512, activation='relu')(_lidar_layer)
    _lidar_layer = tf.keras.layers.Dense(256, activation='relu')(_lidar_layer)
    _lidar_layer = tf.keras.layers.Dropout(drop_prob)(_lidar_layer)
    
    return _input_layer, _lidar_layer


def _image_model(channel=32, drop_prob=0.25):
    _input_layer = tf.keras.layers.Input(shape=(9234, 48, 81))
    _image_layer = tf.keras.layers.concatenate([
        tf.keras.layers.Conv2D(channel, activation='relu', padding='same', kernel_size=(3, 3))(_input_layer),
        tf.keras.layers.Conv2D(channel, activation='relu', padding='same', kernel_size=(7, 7))(_input_layer),
        tf.keras.layers.Conv2D(channel, activation='relu', padding='same', kernel_size=(11, 11))(_input_layer)])

    _image_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(_image_layer)
    _image_layer = tf.keras.layers.Dropout(drop_prob)(_image_layer)
    _image_layer = tf.keras.layers.Conv2D(channel, (3, 3), padding='same', activation='relu')(_image_layer)

    _image_layer = _convolutional_module(_image_layer, channel, drop_prob)
    _image_layer = _convolutional_module(_image_layer, channel, drop_prob)

    _image_layer = tf.keras.layers.Flatten()(_image_layer)
    _image_layer = tf.keras.layers.Dense(512, activation='relu')(_image_layer)
    _image_layer = tf.keras.layers.Dropout(drop_prob)(_image_layer)
    _image_layer = tf.keras.layers.Dense(256, activation='relu')(_image_layer)
    _image_layer = tf.keras.layers.Dropout(drop_prob)(_image_layer)

    return _input_layer, _image_layer


def nu_husky_coord_model_fn():
    coord_input, coord_layer = _coord_model()
    output_layer = tf.keras.layers.Dense(256, activation='softmax')(coord_layer)
    return tf.keras.models.Model(inputs=coord_input, outputs=output_layer)


def nu_husky_lidar_model_fn():
    lidar_input, lidar_layer = _lidar_model()
    output_layer = tf.keras.layers.Dense(256, activation='softmax')(lidar_layer)
    return tf.keras.models.Model(inputs=lidar_input, outputs=output_layer)


def nu_husky_image_model_fn():
    image_input, image_layer = _image_model()
    output_layer = tf.keras.layers.Dense(256, activation='softmax')(image_layer)
    return tf.keras.models.Model(inputs=image_input, outputs=output_layer)


def nu_husky_fusion_model_fn():
    coord_input, coord_layer = _coord_model()
    lidar_input, lidar_layer = _lidar_model()
    image_input, image_layer = _image_model()
    concat_layer = tf.keras.layers.concatenate([coord_layer, lidar_layer, image_layer])

    layer = tf.keras.layers.Dense(512, activation='relu')(concat_layer)
    output_layer = tf.keras.layers.Dense(256, activation='softmax')(layer)
    return tf.keras.models.Model(inputs=[coord_input, image_input, lidar_input], outputs=output_layer)
