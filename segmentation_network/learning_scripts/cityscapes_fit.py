import PIL
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from progress.bar import Bar

from segmentation_network.constants import INPUT_SIZE
from segmentation_network.cnn import UNet

# Label used to denote cars in Cityscapes Dataset.
CAR_LABEL = 26
# Path to images from the training set.
X_PATH_TRAIN = 'data/leftImg8bit_trainvaltest/leftImg8bit/train'
# Path to images from the validation set.
X_PATH_VAL = 'data/leftImg8bit_trainvaltest/leftImg8bit/val'
# Path to ground truths from the training set.
Y_PATH_TRAIN = 'data/gtFine_trainvaltest/gtFine/train'
# Path to ground truths from the validation set.
Y_PATH_VAL = 'data/gtFine_trainvaltest/gtFine/val'
CITIES_TRAIN = ['aachen', 'bochum', 'bremen', 'cologne', 'darmstadt', 'dusseldorf', 'erfurt', 'hamburg', 'hanover',
                'jena', 'krefeld', 'monchengladbach', 'strasbourg', 'stuttgart', 'tubingen', 'ulm', 'weimar', 'zurich']
CITIES_VAL = ['frankfurt', 'lindau', 'munster']


class CityscapesLoader:
    @staticmethod
    def list_files_in_directory(dir_path: str, only_labels: bool = False):
        filenames = sorted([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
        if only_labels:
            filenames = [f for f in filenames if f.endswith('gtFine_labelIds.png')]
        return filenames

    @staticmethod
    def reduce_labels(labels: np.ndarray):
        """
        Transforms a labels image to an array of 0s and 1s (1 means car).
        """
        reduced = np.ndarray(shape=(labels.shape[0], labels.shape[1], 1))
        for x in range(labels.shape[0]):
            for y in range(labels.shape[1]):
                reduced[x, y] = 1 if labels[x, y] == 26 else 0
        return reduced

    @staticmethod
    def random_sample_from_city(city: str):
        if city in CITIES_TRAIN:
            x_path, y_path = X_PATH_TRAIN, Y_PATH_TRAIN
        elif city in CITIES_VAL:
            x_path, y_path = X_PATH_VAL, Y_PATH_VAL
        else:
            assert False, "Invalid city name."
        x_path = os.path.join(x_path, city)
        y_path = os.path.join(y_path, city)
        x_files = CityscapesLoader.list_files_in_directory(x_path)
        y_files = CityscapesLoader.list_files_in_directory(y_path, only_labels=True)
        assert len(x_files) == len(y_files)
        file_idx = random.choice(range(len(x_files)))
        x_path = os.path.join(x_path, x_files[file_idx])
        y_path = os.path.join(y_path, y_files[file_idx])
        x = np.array(PIL.Image.open(x_path).resize(INPUT_SIZE))
        y = np.array(PIL.Image.open(y_path).resize(INPUT_SIZE))
        return x, CityscapesLoader.reduce_labels(y)

    @staticmethod
    def load_city(city: str):
        if city in CITIES_TRAIN:
            x_path, y_path = X_PATH_TRAIN, Y_PATH_TRAIN
        elif city in CITIES_VAL:
            x_path, y_path = X_PATH_VAL, Y_PATH_VAL
        else:
            assert False, "Invalid city name."
        x_path = os.path.join(x_path, city)
        y_path = os.path.join(y_path, city)
        x_files = CityscapesLoader.list_files_in_directory(x_path)
        y_files = CityscapesLoader.list_files_in_directory(y_path, only_labels=True)
        assert len(x_files) == len(y_files)
        xs = []
        ys = []
        bar = Bar("Loading city " + city, max=len(x_files))
        for x_file, y_file in zip(x_files, y_files):
            x_file_path = os.path.join(x_path, x_file)
            y_file_path = os.path.join(y_path, y_file)
            x = np.array(PIL.Image.open(x_file_path).resize(INPUT_SIZE))
            y = np.array(PIL.Image.open(y_file_path).resize(INPUT_SIZE))
            y = CityscapesLoader.reduce_labels(y)
            xs.append(x)
            ys.append(y)
            bar.next()
        bar.finish()
        return np.array(xs), np.array(ys)

    def __init__(self):
        pass


if __name__ == '__main__':
    # x, y = CityscapesLoader.random_sample_from_city('zurich')
    xs, ys = [], []
    for city in CITIES_TRAIN:
        _xs, _ys = CityscapesLoader.load_city(city)
        xs.append(_xs)
        ys.append(_ys)
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    np.save("cityscapes_xs.npy", xs)
    np.save("cityscapes_ys.npy", ys)
    with tf.Session() as sess:
        net = UNet(sess,
                   learning_rate=0.01)
        net.fit(xs, ys, nb_epochs=500)
        net.save()
