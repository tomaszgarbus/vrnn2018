import numpy as np
import PIL
import os
import matplotlib.pyplot as plt

from segmentation_new.cnn import FCN32
from segmentation_new.constants import *
from segmentation_new.learning_scripts.cars_snowball_labels_editor import LabelsEditor
from segmentation_new.cars_loader import CarsLoader


class CarsSnowball:
    @staticmethod
    def store_labels(filename: str, labels: np.ndarray, original_size: tuple):
        print("Storing labels")
        img_arr = np.zeros(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=np.int8)
        for x in range(INPUT_SIZE[0]):
            for y in range(INPUT_SIZE[1]):
                pixl_val = 0 if labels[0, x, y, 0] >= 1 else 255
                img_arr[x, y, 0] = img_arr[x, y, 1] = img_arr[x, y, 2] = pixl_val
        img = PIL.Image.fromarray(img_arr, mode='RGB').resize(original_size)
        img.save(os.path.join(Y_PATH_TRAIN, filename))

    def next(self, net: FCN32):
        global xs, ys
        file = self.unlabeled_files[0]
        x_path = os.path.join(X_PATH_TRAIN, file)
        input_img = PIL.Image.open(x_path)
        img_arr = np.array(input_img.resize(INPUT_SIZE))
        img_arr_normalized = CarsLoader.normalize_pixels(img_arr)
        out = net.predict(np.array([img_arr_normalized]))

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
            elif response == 'n' or response == 'N':
                pass
            else:
                print("Invalid response, assuming \"n\".")
        elif response == 'n' or response == 'N':
            pass
        else:
            print("Invalid response, assuming \"n\".")
        self.unlabeled_files = self.unlabeled_files[1:]

    def __init__(self):
        x_files = CarsLoader.list_files_in_directory(X_PATH_TRAIN)
        y_files = CarsLoader.list_files_in_directory(Y_PATH_TRAIN)
        self.unlabeled_files = list(set(x_files).difference(y_files))


if __name__ == '__main__':
    net = FCN32()
    snowball = CarsSnowball()
    while True:
        snowball.next(net)
