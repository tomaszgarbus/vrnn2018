import numpy as np
import PIL
import os
import matplotlib.pyplot as plt

from segmentation_new.constants import INPUT_SIZE

# Path to the training set from Car dataset.
IMGS_PATH = 'data/cars_train'
# Path to the labels corresponding to the training set of Car dataset.
LABELS_PATH = 'data/cars_train_labels'
# Path to the generated cut images.
CUT_PATH = 'data/cars_train_cut'

xs, ys = None, None


class CarsCutter:
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

    def __init__(self):
        pass

    def process_file(self, filename: str):
        labels_img = PIL.Image.open(os.path.join(LABELS_PATH, filename)).resize(INPUT_SIZE)
        labels_arr = np.array(labels_img)
        labels_arr = self.reduce_labels(labels_arr)

        input_img = PIL.Image.open(os.path.join(IMGS_PATH, filename)).resize(INPUT_SIZE)
        input_arr = np.array(input_img)

        cut_arr = np.array(input_arr * np.tile(labels_arr, input_arr.shape[2]), dtype=input_arr.dtype)
        cut_img = PIL.Image.fromarray(cut_arr, mode=input_img.mode)
        # plt.imshow(cut_img)
        # plt.show()
        cut_img.save(os.path.join(CUT_PATH, filename))

    @staticmethod
    def create_output_dir():
        if not os.path.isdir(CUT_PATH):
            os.makedirs(CUT_PATH)

    def run(self):
        self.create_output_dir()

        label_files = self.list_files_in_directory(LABELS_PATH)
        ct = 0
        for fname in label_files:
            self.process_file(fname)
            ct += 1
            print("%d/%d files processed" % (ct, len(label_files)))


if __name__ == '__main__':
    cutter = CarsCutter()
    cutter.run()