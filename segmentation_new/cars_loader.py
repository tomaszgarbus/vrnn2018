import numpy as np
import PIL
import os

from segmentation_new.constants import *


class CarsLoader:
    @staticmethod
    def list_files_in_directory(dir_path: str):
        filenames = sorted([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
        return filenames


    @staticmethod
    def normalize_pixels(pixels: np.ndarray):
        """
        Subtracts 128 from each pixel.
        """
        normalized = np.ndarray(shape=pixels.shape)
        for x in range(pixels.shape[0]):
            for y in range(pixels.shape[1]):
                for z in range(pixels.shape[2]):
                    normalized[x, y, z] = pixels[x, y, z] - 128.
        return normalized

    @staticmethod
    def reduce_labels(labels: np.ndarray):
        """
        Transforms a labels image to a one hot array (index 1 for car).
        """
        reduced = np.ndarray(shape=(labels.shape[0], labels.shape[1], 2))
        for x in range(labels.shape[0]):
            for y in range(labels.shape[1]):
                tmp = labels[x, y, 0] / 255.
                if tmp < .5:
                    reduced[x, y, 0] = 0.
                    reduced[x, y, 1] = 1.
                else:
                    reduced[x, y, 0] = 1.
                    reduced[x, y, 1] = 0.
        return reduced

    @staticmethod
    def onehot_to_binary(labels: np.ndarray):
        """
        Transforms a one hot array to singular value array(value 1 for car).
        """
        reduced = np.ndarray(shape=(labels.shape[0], labels.shape[1], 1))
        for x in range(labels.shape[0]):
            for y in range(labels.shape[1]):
                reduced[x, y, 0] = 1 if labels[x, y, 1] > labels[x, y, 0] else 0
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
            if x.shape != (INPUT_SIZE[0], INPUT_SIZE[1], 3):
                continue
            x = CarsLoader.normalize_pixels(x)
            y = np.array(PIL.Image.open(y_path).resize(INPUT_SIZE))
            y = CarsLoader.reduce_labels(y)
            xs.append(x.reshape(1, INPUT_SIZE[0], INPUT_SIZE[1], 3))
            ys.append(y.reshape(1, INPUT_SIZE[0], INPUT_SIZE[1], 2))
        ys = np.concatenate(ys)
        return xs, ys

    @staticmethod
    def load_batch_with_no_labels(xs_path, ys_path, batch_size=1000):
        x_files = CarsLoader.list_files_in_directory(xs_path)
        y_files = CarsLoader.list_files_in_directory(ys_path)
        xs = []
        names = []
        sizes = []
        for x_file in x_files:
            if x_file in y_files:
                continue
            x_path = os.path.join(xs_path, x_file)
            x = PIL.Image.open(x_path)
            size = x.size
            x = np.array(x.resize(INPUT_SIZE))
            if x.shape != (INPUT_SIZE[0], INPUT_SIZE[1], 3):
                continue
            x = CarsLoader.normalize_pixels(x)

            xs.append(x.reshape(1, INPUT_SIZE[0], INPUT_SIZE[1], 3))
            names.append(x_file)
            sizes.append(size)

            batch_size -= 1
            if batch_size <= 0:
                break
        xs = np.concatenate(xs) if len(xs) > 0 else np.empty(shape=0)
        return xs, names, sizes

    @staticmethod
    def load_data():
        print("Loading training data...")
        train_x, train_y = CarsLoader.load_set_with_labels(X_PATH_TRAIN, Y_PATH_TRAIN)
        print("Loading validation data...")
        test_x, test_y = CarsLoader.load_set_with_labels(X_PATH_TEST, Y_PATH_TEST)
        print("All data loaded!")
        return train_x, train_y, test_x, test_y

    @staticmethod
    def load_unlabeled_batch(batch_size=1000):
        print("Loading " + str(batch_size) + " pictures...")
        return CarsLoader.load_batch_with_no_labels(X_PATH_PRED, Y_PATH_PRED, batch_size)

    @staticmethod
    def store_labels(pred: bool, filename: str, labels: np.ndarray, original_size: tuple):
        y_path = Y_PATH_PRED if pred else Y_PATH_TRAIN
        img_arr = np.zeros(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=np.int8)
        for x in range(INPUT_SIZE[0]):
            for y in range(INPUT_SIZE[1]):
                pixl_val = 0 if labels[x, y, 0] >= 1 else 255
                img_arr[x, y, 0] = img_arr[x, y, 1] = img_arr[x, y, 2] = pixl_val
        img = PIL.Image.fromarray(img_arr, mode='RGB').resize(original_size)
        img.save(os.path.join(y_path, filename))

    def __init__(self):
        pass

