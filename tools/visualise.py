import matplotlib.pyplot as plt
import numpy as np
import math


def show_images(imgs):
    print("Ruszam, " + str(imgs.shape))
    imgs = (imgs*255).astype(int)
    rows = math.ceil(math.sqrt(len(imgs)))
    cols = np.ceil(len(imgs) / float(rows))
    #plt.imshow(imgs[0, :, :, :])
    #plt.show()
    for i in range(len(imgs)):
        plt.subplot(cols, rows, i+1)
        plt.imshow(imgs[i, :, :, :])
        plt.axis('off')
        #plt.tight_layout()
    plt.show()
