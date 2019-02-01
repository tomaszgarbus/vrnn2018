import numpy as np
import PIL
import os
import matplotlib.pyplot as plt

from segmentation_new.cnn import FCN32
from segmentation_new.constants import *
from segmentation_new.learning_scripts.cars_snowball_labels_editor import LabelsEditor
from segmentation_new.cars_loader import CarsLoader


class CarsSnowball:

    def next(self, net: FCN32):
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
            print("Storing labels...")
            CarsLoader.store_labels(False, file, out[0], input_img.size[:2])
        elif response == 'c' or response == 'C':
            editor = LabelsEditor(img_arr, out)
            plt.imshow(np.reshape(out, INPUT_SIZE))
            plt.show()
            print("Save this label? [y/N]")
            response = input().strip()
            if response == 'y' or response == 'Y':
                print("Storing labels...")
                CarsLoader.store_labels(False, file, out[0], input_img.size[:2])
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
