import os
import tempfile
import zipfile
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from core.dataset import output_dataset
from core.metrics import TopKThroughputRatio
from models import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_fn, dataset_fn = models['imperial']
training_input, validation_input = dataset_fn()
training_output, validation_output = output_dataset()

model = tf.keras.models.load_model('../results/models/centralised-imperial/model',
                                   custom_objects={'TopKThroughputRatio': TopKThroughputRatio})
model.evaluate(validation_input, validation_output)

# Define model for pruning
pruning_epochs = 2
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50, final_sparsity=0.80, begin_step=0,
                                                             end_step=len(validation_input) * pruning_epochs)
}

# The accuracy and throughput metrics
metrics = [
    tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top-1-accuracy'),
    tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top-10-accuracy'),
    TopKThroughputRatio(k=1, name='top-1-throughput'),
    TopKThroughputRatio(k=10, name='top-10-throughput')
]


def apply_layer_pruning(layer):
    return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params) \
        if isinstance(layer, tf.keras.layers.Dense) else layer


pruning_model = tf.keras.models.clone_model(model, clone_function=apply_layer_pruning)
pruning_model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=metrics)

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir='tmp'),
]

pruning_model.fit(training_input, training_output, batch_size=16, epochs=2, verbose=2,
                  validation_data=(validation_input, validation_output), callbacks=callbacks)


def gzipped_model_size(_model):
    """
    Calculates the GZipped size of the model

    :param _model: the model
    :return: GZipped size of the model
    """
    _, keras_file = tempfile.mkstemp('.h5')
    _model.save(keras_file, include_optimizer=False)

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(keras_file)

    return os.path.getsize(zipped_file)


print(f"Size of gzipped pruned model without stripping: {gzipped_model_size(pruning_model):.2f} bytes")
stripped_pruning_model = tfmot.sparsity.keras.strip_pruning(pruning_model)
print(f"Size of gzipped pruned model with stripping: {gzipped_model_size(stripped_pruning_model):.2f} bytes")
