import argparse
import json
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
]

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    input_data = np.array(data["mfcc"])
    target_data = np.array(data["labels"])
    mapping = np.array(data["mapping"])

    return input_data, target_data, mapping


def prepare_dataset(args):
    X, Y, labels = load_data(args.dataset_path)

    # Create train/test split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=args.test_size)

    # Create train/validation split
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=args.validation_size)

    print('Training labels shape:', y_train.shape)
    print('Validation labels shape:', y_validation.shape)
    print('Test labels shape:', y_test.shape)

    print('Training features shape:', x_train.shape)
    print('Validation features shape:', x_validation.shape)
    print('Test features shape:', x_test.shape)

    initial_bias = calculate_initial_bias(target_data=Y)
    class_weight = calculate_class_weight(target_data=Y)
    return x_train, x_validation, x_test, y_train, y_validation, y_test, labels, initial_bias, class_weight


def calculate_initial_bias(target_data):
    neg, pos = np.bincount(target_data)

    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))
    return np.log([pos / neg])


def calculate_class_weight(target_data):
    neg, pos = np.bincount(target_data)

    total = neg + pos
    weight_for_0 = (1 / neg) * total / 2.0
    weight_for_1 = (1 / pos) * total / 2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    return class_weight


def make_model(shape, output_bias=None):
    # Create model
    mdl = keras.Sequential()
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)
        # 1st LSTM layer
        mdl.add(keras.layers.LSTM(128, input_shape=shape, return_sequences=True))

        # 2nd LSTM layer
        mdl.add(keras.layers.LSTM(128))

        # Dense layer
        mdl.add(keras.layers.Dense(128, activation='relu'))
        mdl.add(keras.layers.Dropout(0.3))

        # Output layer
        mdl.add(keras.layers.Dense(2, activation='sigmoid', bias_initializer=output_bias))
    return mdl


def plot_metrics(history):
    metrics = ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=COLORS[0], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color=COLORS[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()


def main(args):
    # Split the data into train and test datasets
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test, labels, initial_bias, class_weight = prepare_dataset(
        args)

    # Build the RNN-LSTM
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = make_model(input_shape, output_bias=initial_bias)

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    loss = keras.losses.BinaryCrossentropy
    model.compile(optimizer=optimizer, loss=loss, metrics=METRICS)

    initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
    model.save_weights(initial_weights)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_auc',
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)

    model.summary()
    history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), batch_size=args.batch_size,
                        epochs=args.epochs, callbacks=early_stopping, class_weight=class_weight)

    plot_metrics(history)

    results = model.evaluate(X_train, Y_train, batch_size=args.batch_size, verbose=0)
    print("Loss: {:0.4f}\n"
          "Accuracy: {:0.4f}".format(results[0], results[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        This Script will split any audio file passed in parameter into chunks of a specified length
        """)

    parser.add_argument("--dataset_path", type=str, help="This is the path to the directory where the datasets reside",
                        required=True)
    parser.add_argument("--test_size", type=float, default=0.25, help="Size of the test dataset")
    parser.add_argument("--validation_size", type=float, default=0.2, help="Size of the validation dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")

    args = parser.parse_args()
    main(args)
