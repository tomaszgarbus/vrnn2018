from PIL import Image, ImageOps
import numpy as np
import PIL
import os
import os.path
import matplotlib.pyplot as plt
from segmentation_network.learning_scripts.cars_fit import CarsLoader
from itertools import product
import pickle
from tqdm import tqdm
from tools.visualise import show_images

X_PATH_TRAIN = 'data/cars_train'
Y_PATH_TRAIN = 'data/cars_train_labels'
X_SIZE = 32
Y_SIZE = 32
CACHE_FILE = "gan_prep_cache_32"
BACKGROUND_COLOR = 0  # 255 = white


class Preprocess:

    def __init__(self,
                 x_size = X_SIZE,
                 y_size = Y_SIZE,
                 x_path=X_PATH_TRAIN,
                 y_path=Y_PATH_TRAIN,
                 background_color=BACKGROUND_COLOR,
                 cache_file=CACHE_FILE):
        self.background_color = background_color
        self.x_path = x_path
        self.y_path = y_path
        self.x_size = x_size
        self.y_size = y_size
        self.cache_file = cache_file
        self.dataset = self.try_loading()

    def pad_and_augment_image(self, image):
        pimg = Image.fromarray(image)
        old_size = pimg.size
        ratio = min(self.x_size / old_size[0], self.y_size / old_size[1])
        new_size = tuple([round(x * ratio) for x in old_size])
        pimg = pimg.resize(new_size, Image.ANTIALIAS)
        assert pimg.size[0] <= self.x_size and pimg.size[1] <= self.y_size

        delta_w = self.x_size - new_size[0]
        delta_h = self.y_size - new_size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        new_img = ImageOps.expand(pimg, padding, fill=tuple(3*[self.background_color]))
        x = np.array(new_img).astype(np.float16) / 255
        assert x.shape == (self.x_size, self.y_size, 3)
        flipped = np.fliplr(x).reshape(1, self.x_size, self.y_size, 3)
        x = x.reshape(1, self.x_size, self.y_size, 3)
        return [x, flipped]  # może coś wiecej?

    def save(self):
        with open(self.cache_file, 'wb') as output:
            pickle.dump(self.dataset, output)

    def try_loading(self):
        if os.path.isfile(self.cache_file):
            with open(self.cache_file, 'rb') as inp:
                return pickle.load(inp)
        if os.path.isfile(self.cache_file + '.npy'):
            return np.load(self.cache_file + '.npy')
        return None

    def cut_car(self, image, mask):
        shape = image.shape
        assert shape[0] == mask.shape[0] and shape[1] == mask.shape[1]
        for (x, y, z) in product(*[range(shape[0]), range(shape[1]), range(shape[2])]):
            image[x, y, z] = image[x, y, z] if mask[x, y] > .5 else self.background_color
        return image

    def load_dataset(self):
        if self.dataset is not None:
            return self.dataset
        x_files = CarsLoader.list_files_in_directory(X_PATH_TRAIN)
        y_files = CarsLoader.list_files_in_directory(Y_PATH_TRAIN)
        cars = []
        for y_file in tqdm(y_files, desc="Preprocessing files"):
            assert y_file in x_files
            x_path = os.path.join(X_PATH_TRAIN, y_file)
            y_path = os.path.join(Y_PATH_TRAIN, y_file)
            x = np.array(Image.open(x_path))
            y = np.array(Image.open(y_path))
            y = CarsLoader.reduce_labels(y)
            cut = self.cut_car(x, y)

            new_car = self.pad_and_augment_image(cut)
            #show_images(np.concatenate(2*new_car))
            cars = cars + new_car
        self.dataset = np.concatenate(cars)
        self.save()
        return self.dataset
