import tensorflow as tf
import sys
import PIL
import numpy as np
import matplotlib.pyplot as plt

from segmentation_network.cnn import UNet
from segmentation_network.constants import INPUT_SIZE


def predict_and_show(net: UNet, file: str):
    input_img = PIL.Image.open(file)
    img_arr = np.array(input_img.resize(INPUT_SIZE))
    out = net.predict(np.array([img_arr]))
    img_arr_with_filter = img_arr.copy()
    for x in range(INPUT_SIZE[0]):
        for y in range(INPUT_SIZE[1]):
            img_arr_with_filter[x, y, 0] = min(img_arr_with_filter[x, y, 0] + 90 * out[0, x, y, 0],
                                               255)
    plt.imshow(img_arr_with_filter)
    plt.show()
    # plt.imshow(np.reshape(out, INPUT_SIZE))
    # plt.show()
    return out


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Provide input file name.")
        exit(0)

    with tf.Session() as sess:
        net = UNet(sess)
        predict_and_show(net, sys.argv[1])
