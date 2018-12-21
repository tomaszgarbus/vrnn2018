import tensorflow as tf
import sys
import PIL
import numpy as np
import matplotlib.pyplot as plt

from segmentation_network.cnn import UNet
from segmentation_network.constants import INPUT_SIZE

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Provide input file name.")
        exit(0)

    input_img = PIL.Image.open(sys.argv[1])
    img_arr = np.array(input_img.resize(INPUT_SIZE))

    with tf.Session() as sess:
        net = UNet(sess)
        out = net.predict(np.array([img_arr]))
        plt.imshow(np.reshape(out, INPUT_SIZE))
        plt.show()
