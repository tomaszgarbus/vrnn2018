import tensorflow as tf
import numpy as np
import PIL
import os
import matplotlib.pyplot as plt

from segmentation_network.cnn import UNet
from segmentation_network.constants import INPUT_SIZE

# Path to the training set from Car dataset.
X_PATH_TRAIN = 'data/cars_train'
# Path to the labels corresponding to the training set of Car dataset.
Y_PATH_TRAIN = 'data/cars_train_labels'


class CarsLoader:
    @staticmethod
    def list_files_in_directory(dir_path: str):
        filenames = sorted([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
        return filenames

    @staticmethod
    def reduce_labels(labels: np.ndarray):
        """
        Transforms a labels image to an array of 0s and 1s (1 means car).
        """
        reduced = np.ndarray(shape=(labels.shape[0], labels.shape[1], 1))
        for x in range(labels.shape[0]):
            for y in range(labels.shape[1]):
                tmp = labels[x, y, 0] / 255.
                reduced[x, y] = 1 if tmp < .5 else 0
        return reduced

    @staticmethod
    def load_set_with_labels(xs_path, ys_path):
        x_files = CarsLoader.list_files_in_directory(xs_path)
        y_files = CarsLoader.list_files_in_directory(ys_path)
        xs = []
        ys = []
        for y_file in y_files:
            assert y_file in x_files
            x_path = os.path.join(xs_path, y_file)
            y_path = os.path.join(ys_path, y_file)
            x = np.array(PIL.Image.open(x_path).resize(INPUT_SIZE))
            y = np.array(PIL.Image.open(y_path).resize(INPUT_SIZE))
            y = CarsLoader.reduce_labels(y)
            xs.append(x.reshape(1, INPUT_SIZE[0], INPUT_SIZE[1], 3))
            ys.append(y.reshape(1, INPUT_SIZE[0], INPUT_SIZE[1], 1))
        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        return xs, ys


    @staticmethod
    def load_training_set_with_labels():
        return CarsLoader.load_set_with_labels(X_PATH_TRAIN, Y_PATH_TRAIN)

    def __init__(self):
        pass


if __name__ == '__main__':
    xs, ys = CarsLoader.load_training_set_with_labels()
    with tf.Session() as sess:
        net = UNet(sess,
                   learning_rate=0.0001)
        net.fit(xs, ys, nb_epochs=500)
        net.save()
