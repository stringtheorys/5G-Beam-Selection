
import tensorflow as tf


@tf.function
def training_step(model, x, y, loss_fn, optimiser, metrics):
    """
    Training step

    :param model: keras model
    :param x: training input
    :param y: training output
    :param loss_fn: categorical loss function
    :param optimiser: keras optimiser
    :param metrics: dictionary of metrics
    :return: the error from the loss function
    """
    # Model prediction
    with tf.GradientTape() as tape:
        predicted = model(x, training=True)
        loss = loss_fn(predicted, y)

    # Backpropagation
    gradients = tape.gradient(loss, model.trainable_variables)
    optimiser.apply_gradients(zip(gradients, model.trainable_variables))

    # Update the metrics for the training
    for metric in metrics:
        metric.update_state(y, predicted)

    return loss


@tf.function
def validation_step(model, x, y, metrics):
    predicted = model(x, training=False)

    for metric in metrics:
        metric.update_state(y, predicted)
