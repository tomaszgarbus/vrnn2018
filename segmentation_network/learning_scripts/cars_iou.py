import tensorflow as tf
import numpy as np
import PIL
import os
import matplotlib.pyplot as plt

from segmentation_network.cnn import UNet
from segmentation_network.constants import *
from segmentation_network.learning_scripts.cars_snowball import CarsLoader


if __name__ == '__main__':
    with tf.Session() as sess:
        net = UNet(sess, learning_rate=0.0001)

        xs, ys = CarsLoader.load_set_with_labels(X_PATH_TEST, Y_PATH_TEST)

        ys = np.array(ys, dtype='int')
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

        for i in range(iou.shape[0]):
            print("IoU " + str(i) + ": " + str(iou[i]))
        print("mean IoU: " + str(iou.mean()))
