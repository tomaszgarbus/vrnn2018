import tensorflow as tf
import numpy as np
import PIL
import os
import matplotlib.pyplot as plt

from segmentation_network.cnn import UNet
from segmentation_network.constants import INPUT_SIZE
from segmentation_network.learning_scripts.cars_fit import CarsLoader
from segmentation_network.learning_scripts.cars_snowball_labels_editor import LabelsEditor

# Path to the training set from Car dataset.
X_PATH_TRAIN = 'data/cars_train'
# Path to the labels corresponding to the training set of Car dataset.
Y_PATH_TRAIN = 'data/cars_train_labels'


xs, ys = None, None


class CarsSnowball:
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
    def store_labels(filename: str, labels: np.ndarray, original_size: tuple):
        img_arr = np.zeros(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=np.int8)
        for x in range(INPUT_SIZE[0]):
            for y in range(INPUT_SIZE[1]):
                pixl_val = 0 if labels[0, x, y, 0] >= 1 else 255
                img_arr[x, y, 0] = img_arr[x, y, 1] = img_arr[x, y, 2] = pixl_val
        img = PIL.Image.fromarray(img_arr, mode='RGB').resize(original_size)
        img.save(os.path.join(Y_PATH_TRAIN, filename))

    def next(self, net: UNet):
        global xs, ys
        file = self.unlabeled_files[0]
        x_path = os.path.join(X_PATH_TRAIN, file)
        input_img = PIL.Image.open(x_path)
        img_arr = np.array(input_img.resize(INPUT_SIZE))
        out = net.predict(np.array([img_arr]))

        img_arr_with_filter = img_arr.copy()
        for x in range(INPUT_SIZE[0]):
            for y in range(INPUT_SIZE[1]):
                nval = img_arr_with_filter[x, y, 0] + 90 * (out[0, x, y, 0] - .3)
                img_arr_with_filter[x, y, 0] = max(0, min(255, nval))
        plt.imshow(img_arr_with_filter)
        plt.show()

        print("Was this car labeled correctly? [y/c/N]")
        response = input().strip()
        if response == 'y' or response == 'Y':
            CarsSnowball.store_labels(file, out, input_img.size[:2])
            xs = np.concatenate([xs, np.array([img_arr])])
            ys = np.concatenate([ys, out])
        elif response == 'c' or response == 'C':
            editor = LabelsEditor(img_arr, out)
            plt.imshow(np.reshape(out, INPUT_SIZE))
            plt.show()
            print("Save this label? [y/N]")
            response = input().strip()
            if response == 'y' or response == 'Y':
                CarsSnowball.store_labels(file, out, input_img.size[:2])
                xs = np.concatenate([xs, np.array([img_arr])])
                ys = np.concatenate([ys, out])
            else:
                pass
        elif response == 'n' or response == 'N':
            pass
        else:
            print("Invalid response, assuming \"n\".")
        self.unlabeled_files = self.unlabeled_files[1:]

    def __init__(self):
        x_files = CarsSnowball.list_files_in_directory(X_PATH_TRAIN)
        y_files = CarsSnowball.list_files_in_directory(Y_PATH_TRAIN)
        self.unlabeled_files = list(set(x_files).difference(y_files))


if __name__ == '__main__':
    with tf.Session() as sess:
        xs, ys = CarsLoader.load_training_set_with_labels()

        net = UNet(sess, learning_rate=0.0001)
        snowball = CarsSnowball()
        while True:
            snowball.next(net)
            net.fit(xs, ys, nb_epochs=1)
    pass
