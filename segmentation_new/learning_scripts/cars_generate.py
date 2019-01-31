import numpy as np
import PIL
import os
import matplotlib.pyplot as plt

from segmentation_new.cnn import FCN32
from segmentation_new.constants import *
from segmentation_new.cars_loader import CarsLoader


if __name__ == '__main__':
    net = FCN32()
    while True:
        input, names, sizes = CarsLoader.load_unlabeled_batch(batch_size=1000)
        if len(input) == 0:
            break
        output = net.predict(input)
        for i in range(output.shape[0]):
            CarsLoader.store_labels(True, names[i], output[i], sizes[i])

