import collections

import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_model_optimization as tfmot

from dataloader import LidarDataset2D
import os
import zipfile
import tempfile
import pruningToolkit
import time

print("paranoia check")

# Arguments
MONTECARLO = 1
NUM_VEHICLES = 2
LOCAL_EPOCHS = 1
AGGREGATION_ROUNDS = 1

BATCH_SIZE = 16
SHUFFLE_BUFFER = 20
PREFETCH_BUFFER = 10

lidar_training_path = ["lidar_input_train.npz", "lidar_input_validation.npz"]  # Raymobtime s008
beam_training_path = ["beams_output_train.npz", "beams_output_validation.npz"]  # Raymobtime s008

lidar_test_path = ["lidar_input_test.npz"]  # Raymobtime s009
beam_test_path = ["beams_output_test.npz"]  # Raymobtime s009

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, 0, frequency=100),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}


# Functions
def get_local_dataset(lidar_path, beam_path, num_vehicles, vehicle_ID):
    training_data = LidarDataset2D(lidar_path, beam_path)
    training_data.lidar_data = np.transpose(training_data.lidar_data, (0, 2, 3, 1))
    x = training_data.lidar_data
    # Split Lidar Data
    xx = x[vehicle_ID * int(x.shape[0] / num_vehicles):(vehicle_ID + 1) * int(x.shape[0] / num_vehicles), :, :, :]
    y = training_data.beam_output
    # Split Beam Labels
    yy = y[vehicle_ID * int(y.shape[0] / num_vehicles):(vehicle_ID + 1) * int(y.shape[0] / num_vehicles), :]

    dataset_train = tf.data.Dataset.from_tensor_slices((list(xx.astype(np.float32)), list(yy.astype(np.float32))))
    # sio.savemat('label'+str(k)+'.mat',{'label'+str(k):yy})
    return dataset_train


def get_test_dataset(lidar_path, beam_path):
    test_data = LidarDataset2D(lidar_path, beam_path)
    test_data.lidar_data = np.transpose(test_data.lidar_data, (0, 2, 3, 1))
    dataset_test = tf.data.Dataset.from_tensor_slices(
        (list(test_data.lidar_data.astype(np.float32)), list(test_data.beam_output.astype(np.float32))))
    return dataset_test


def preprocess(dataset):
    def batch_format_fn(element1, element2):
        return collections.OrderedDict(x=element1, y=element2)

    return dataset.repeat(LOCAL_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).map(batch_format_fn).prefetch(
        PREFETCH_BUFFER)


def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(20, 200, 1)),
        tf.keras.layers.Conv2D(5, 3, 1, padding='same'),  # kernel_initializer=initializers.HeUniform),
        tf.keras.layers.BatchNormalization(axis=3),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
        tf.keras.layers.Conv2D(5, 3, 1, padding='same'),  # kernel_initializer=initializers.HeUniform),
        tf.keras.layers.BatchNormalization(axis=3),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
        tf.keras.layers.Conv2D(5, 3, 2, padding='same'),  # kernel_initializer=initializers.HeUniform),
        tf.keras.layers.BatchNormalization(axis=3),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
        tf.keras.layers.Conv2D(5, 3, 1, padding='same'),  # kernel_initializer=initializers.HeUniform),
        tf.keras.layers.BatchNormalization(axis=3),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
        tf.keras.layers.Conv2D(5, 3, 2, padding='same'),  # , kernel_initializer=initializers.HeUniform),
        tf.keras.layers.BatchNormalization(axis=3),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
        tf.keras.layers.Conv2D(1, 3, (1, 2), padding='same'),  # , kernel_initializer=initializers.HeUniform),
        tf.keras.layers.BatchNormalization(axis=3),
        tf.keras.layers.PReLU(shared_axes=[1, 2]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16),
        tf.keras.layers.ReLU(),
        # layers.Dropout(0.7),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Softmax()])


temp_dataset = get_local_dataset(lidar_training_path, beam_training_path, NUM_VEHICLES, 0)
preprocessed_example_dataset = preprocess(temp_dataset)
example_element = next(iter(preprocessed_example_dataset))


def model_fn():
    keras_model = create_keras_model()
    top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_categorical_accuracy')
    top10 = tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10_categorical_accuracy')
    return tff.learning.from_keras_model(keras_model,
                                         input_spec=preprocessed_example_dataset.element_spec,
                                         loss=tf.keras.losses.CategoricalCrossentropy(),
                                         metrics=[top1, top10])


def apply_pruning_to_dense(layer):
    if isinstance(layer, tf.keras.layers.Dense):
        return tfmot.sparsity.keras.prune_low_magnitude(layer,**pruning_params)
    return layer

def get_gzipped_model_size(model):
    _, new_pruned_keras_file = tempfile.mkstemp(".h5")
    print("Saving pruned model to: ", new_pruned_keras_file)
    tf.keras.models.save_model(model, new_pruned_keras_file, include_optimizer=False)
    
    # Zip the .h5 model file
    _, zip3 = tempfile.mkstemp(".zip")
    with zipfile.ZipFile(zip3, "w", compression=zipfile.ZIP_DEFLATED) as f:
        f.write(new_pruned_keras_file)
    print(
        "Size of ",str(model)," before compression: %.5f Mb"
        % (os.path.getsize(new_pruned_keras_file) / float(2 ** 20))
    )
    print(
        "Size of ",str(model)," after compression: %.5f Mb"
        % (os.path.getsize(zip3) / float(2 ** 20))
    )

def num_pruned(model):
    for i, w in enumerate(model.get_weights()):
        print(
            "{} -- Total:{}, Zeros: {:.2f}%".format(
                model.weights[i].name, w.size, np.sum(w == 0) / w.size * 100
            )
        )

def testStuff(modelInput):
    #print(modelInput)
    print('sparsity = ',pruningToolkit.currentSparcity(modelInput.weights[0]))
    test_preds = modelInput.predict(test_data.lidar_data, batch_size=100)
    test_preds_idx = np.argsort(test_preds, axis=1)
    top_k = np.zeros(100)
    throughput_ratio_at_k = np.zeros(100)
    correct = 0
    for i in range(100):
        correct += np.sum(test_preds_idx[:, -1 - i] == np.argmax(test_data.beam_output, axis=1))
        top_k[i] = correct / test_data.beam_output.shape[0]
        throughput_ratio_at_k[i] = np.sum(np.log2(
            np.max(np.take_along_axis(test_data.beam_output_true, test_preds_idx, axis=1)[:, -1 - i:],
                    axis=1) + 1.0)) / np.sum(np.log2(np.max(test_data.beam_output_true, axis=1) + 1.0))
    #accFL = metrics['train']['top_10_categorical_accuracy'] / MONTECARLO
    
    #print('acc = ',throughput_ratio_at_k)
    
    
    
    
# Main
iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(lr=5e-3),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(lr=.25))

evaluation = tff.learning.build_federated_evaluation(model_fn)

test_data = LidarDataset2D(lidar_test_path, beam_test_path)
test_data.lidar_data = np.transpose(test_data.lidar_data, (0, 2, 3, 1))

accFL = 0
for MONTECARLOi in range(MONTECARLO):
    # Generate Federated Train Dataset
    federated_train_data = []
    for i in range(NUM_VEHICLES):
        train_dataset = get_local_dataset(lidar_training_path, beam_training_path, NUM_VEHICLES, i)
        federated_train_data.append(preprocess(train_dataset))

    # Generate Test Dataset
    test_dataset = get_test_dataset(lidar_test_path, beam_test_path)
    federated_test_data = [preprocess(test_dataset)]

    top1 = np.zeros(AGGREGATION_ROUNDS)
    top10 = np.zeros(AGGREGATION_ROUNDS)

    state = iterative_process.initialize()  # Initialize training

    # top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_categorical_accuracy', dtype=None)
    # top10 = tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10_categorical_accuracy', dtype=None)

    # Federated Training
    for round_num in range(AGGREGATION_ROUNDS):
        state, metrics = iterative_process.next(state, federated_train_data)
        test_metrics = evaluation(state.model, federated_test_data)

        #print(str(metrics),"\n")
        #print(str(test_metrics),3*"\n")

        top1[round_num] = test_metrics['top_1_categorical_accuracy']
        top10[round_num] = test_metrics['top_10_categorical_accuracy']

        # Generate Accuracy and Throughput Performance Curves
        keras_model = create_keras_model()
        # keras_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[top1,top10])
        state.model.assign_weights_to(keras_model)
        
        #print(state.model)
        
        #print("Base Model Summary",2*"\n")
        #keras_model.summary()
        # print('Base Model:','\n')
        # get_gzipped_model_size(keras_model)
        
        # a = time.perf_counter()
        # model_for_pruning = pruningToolkit.pruneModel(keras_model, 0.5)
        # b = time.perf_counter()
        # print('Pruned Model:','\n')
        # #get_gzipped_model_size(model_for_pruning)
        # print('Eval time = ', b - a)
        
        #keras_model.summary()
        #model_for_pruning.summary()
        
        
        
        XSparce = pruningToolkit.pruneModel(keras_model, 0.1)
        testStuff(XSparce)
        # XX = pruningToolkit.pruneModel(keras_model, 0.2)
        #get_gzipped_model_size(XX)
        # testStuff(XX)
        # XXX = pruningToolkit.pruneModel(keras_model, 0.3)
        # testStuff(XXX)
        #get_gzipped_model_size(XXX)
        # XL = pruningToolkit.pruneModel(keras_model, 0.4)
        # testStuff(XL)
        #get_gzipped_model_size(XL)
        # L = pruningToolkit.pruneModel(keras_model, 0.5)
        # testStuff(L)
        #get_gzipped_model_size(L)
        # LX = pruningToolkit.pruneModel(keras_model, 0.6)
        # testStuff(LX)
        #get_gzipped_model_size(LX)
        # LXX = pruningToolkit.pruneModel(keras_model, 0.7)
        # testStuff(LXX)
        #get_gzipped_model_size(LXX)
        # LXXX = pruningToolkit.pruneModel(keras_model, 0.8)
        # testStuff(LXXX)
        #get_gzipped_model_size(LXXX)
        # XC = pruningToolkit.pruneModel(keras_model, 0.9)
        # testStuff(XC)
        #get_gzipped_model_size(XC)

        # testStuff(keras_model)
        
        # model_for_pruning = tf.keras.models.clone_model(keras_model,clone_function=apply_pruning_to_dense)
                                                        
        # print(2*"\n","Pruned Model Summary",2*"\n")
        # model_for_pruning.summary()
        # model_for_pruning.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[top1,top10])
        # num_pruned(model_for_pruning)
        # get_gzipped_model_size(model_for_pruning)
        
        # model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

        # print(2*"\n","Strip Pruned Model Summary",2*"\n")
        # model_for_export.summary()
        # num_pruned(model_for_export)
        # get_gzipped_model_size(model_for_export)
        
        # test_preds = model_for_pruning.predict(test_data.lidar_data, batch_size=100,callbacks = tfmot.sparsity.keras.UpdatePruningStep())
        # test_preds_idx = np.argsort(test_preds, axis=1)
        # top_k = np.zeros(100)
        # throughput_ratio_at_k = np.zeros(100)
        # correct = 0
        # for i in range(100):
        #     correct += np.sum(test_preds_idx[:, -1 - i] == np.argmax(test_data.beam_output, axis=1))
        #     top_k[i] = correct / test_data.beam_output.shape[0]
        #     throughput_ratio_at_k[i] = np.sum(np.log2(
        #         np.max(np.take_along_axis(test_data.beam_output_true, test_preds_idx, axis=1)[:, -1 - i:],
        #                axis=1) + 1.0)) / np.sum(np.log2(np.max(test_data.beam_output_true, axis=1) + 1.0))

    #     sio.savemat('federated_accuracy' + str(round_num) + '.mat', {'accuracy': top_k})
    #     sio.savemat('federated_throughput' + str(round_num) + '.mat', {'throughput': throughput_ratio_at_k})

    # sio.savemat('top1.mat', {'top1': top1})
    # sio.savemat('top10.mat', {'top10': top10})

    # np.savez("federated.npz", classification=top_k, throughput_ratio=throughput_ratio_at_k)
    # accFL = accFL + metrics['train']['top_10_categorical_accuracy'] / MONTECARLO

#     print(MONTECARLOi,"\n")

# print(accFL)





