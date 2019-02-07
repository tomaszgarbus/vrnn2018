import numpy as np
import PIL
import os
import matplotlib.pyplot as plt

from segmentation_new.cnn import FCN32
from segmentation_new.constants import *
from segmentation_new.cars_loader import CarsLoader

def mean_iou(xs: np.ndarray, ys: np.ndarray):
    bin_ys = []
    for i in range(ys.shape[0]):
        bin_ys.append(CarsLoader.onehot_to_binary(ys[i]))

    ys = np.array(bin_ys, dtype='int')
    ys = ys.reshape((ys.shape[0], INPUT_SIZE[0] * INPUT_SIZE[1]))

    preds = np.array(net.predict(xs), dtype='int')
    preds = preds.reshape((preds.shape[0], INPUT_SIZE[0] * INPUT_SIZE[1]))
    inter = ys & preds
    union = ys | preds

    inter_sums = inter.sum(axis=1, dtype='double')
    union_sums = union.sum(axis=1, dtype='double')

    # if no car ground truth then empty match is perfect
    for i in range(union_sums.shape[0]):
        if union_sums[i] == 0.:
            inter_sums[i] = 1.
            union_sums[i] = 1.

    iou = inter_sums / union_sums
    return iou.mean()



if __name__ == '__main__':
    net = FCN32()
    val_xs, val_ys = CarsLoader.load_set_with_labels(X_PATH_TEST, Y_PATH_TEST)
    test_xs, test_ys = CarsLoader.load_set_with_labels(X_PATH_BENCH, Y_PATH_BENCH)

    print("Validation set mean IoU: " + str(mean_iou(val_xs, val_ys)))
    print("Test set mean IoU: " + str(mean_iou(test_xs, test_ys)))
