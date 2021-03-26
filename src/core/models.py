
import tensorflow as tf


imperial_model = tf.keras.models.Sequential([
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


def beamsoup_lidar_layers():
    _lidar_input = tf.keras.layers.Input(shape=(20, 200, 1))
    _lidar_layer = tf.math.divide(tf.add(tf.cast(_lidar_input, dtype=tf.float32), 2), 3)  # Scaling to [0,1] interval
    _lidar_layer = tf.keras.layers.GaussianNoise(0.005)(_lidar_layer)  # 0.002
    _lidar_layer = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same')(_lidar_layer)
    _lidar_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(_lidar_layer)
    _lidar_layer = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same')(_lidar_layer)
    _lidar_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(_lidar_layer)
    _lidar_layer = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same')(_lidar_layer)
    _lidar_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(_lidar_layer)
    _lidar_layer = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(_lidar_layer)
    _lidar_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(_lidar_layer)
    _lidar_layer = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(_lidar_layer)
    _lidar_layer = tf.keras.layers.Flatten()(_lidar_layer)
    _lidar_layer = tf.keras.layers.Dense(400, activation='relu', kernel_regularizer='l2')(_lidar_layer)
    return _lidar_input, _lidar_layer


def beamsoup_coord_layers():
    # Coord model
    _coord_input = tf.keras.layers.Input(shape=(2,))
    _coord_layer = tf.keras.layers.Dense(128, activation='relu')(_coord_input)
    _coord_layer = tf.keras.layers.GaussianNoise(0.002)(_coord_layer)
    return _coord_input, _coord_layer


# Lidar model
lidar_input, lidar_layer = beamsoup_lidar_layers()
output_layer = tf.keras.layers.Dense(256, activation='softmax')(lidar_layer)
beamsoup_lidar_model = tf.keras.models.Model(inputs=lidar_input, outputs=output_layer)

# Coord model
coord_input, coord_layer = beamsoup_coord_layers()
output_layer = tf.keras.layers.Dense(256, activation='softmax')(coord_layer)
beamsoup_coord_model = tf.keras.models.Model(inputs=coord_input, outputs=output_layer)

# Combine the coord and lidar models
lidar_input, lidar_layer = beamsoup_lidar_layers()
coord_input, coord_layer = beamsoup_coord_layers()
alignment_layer = tf.keras.layers.concatenate([lidar_layer, coord_layer])
alignment_layer = tf.keras.layers.Dense(600, activation='relu')(alignment_layer)
alignment_layer = tf.keras.layers.Dense(600, activation='relu')(alignment_layer)
alignment_layer = tf.keras.layers.Dense(500, activation='relu')(alignment_layer)
alignment_layer = tf.keras.layers.Dense(256, activation='softmax')(alignment_layer)
beamsoup_lidar_coord_model = tf.keras.models.Model(inputs=[lidar_input, coord_input], outputs=alignment_layer)
