import tensorflow as tf


def imperial_model_fn():
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(20, 200, 1)),
        tf.keras.layers.Conv2D(5, 3, 1, padding='same', kernel_initializer=tf.keras.initializers.HeUniform()),
        tf.keras.layers.BatchNormalization(axis=3),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
        tf.keras.layers.Conv2D(5, 3, 1, padding='same', kernel_initializer=tf.keras.initializers.HeUniform()),
        tf.keras.layers.BatchNormalization(axis=3),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
        tf.keras.layers.Conv2D(5, 3, 2, padding='same', kernel_initializer=tf.keras.initializers.HeUniform()),
        tf.keras.layers.BatchNormalization(axis=3),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
        tf.keras.layers.Conv2D(5, 3, 1, padding='same', kernel_initializer=tf.keras.initializers.HeUniform()),
        tf.keras.layers.BatchNormalization(axis=3),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
        tf.keras.layers.Conv2D(5, 3, 2, padding='same', kernel_initializer=tf.keras.initializers.HeUniform()),
        tf.keras.layers.BatchNormalization(axis=3),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
        tf.keras.layers.Conv2D(1, 3, (1, 2), padding='same', kernel_initializer=tf.keras.initializers.HeUniform()),
        tf.keras.layers.BatchNormalization(axis=3),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),  # This was originally 16
        # layers.Dropout(0.7),
        tf.keras.layers.Dense(256, activation='softmax')
    ])
