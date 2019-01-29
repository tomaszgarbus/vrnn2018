import matplotlib.pyplot as plt
import numpy as np
import math
import os
import datetime


def show_images(imgs, title="", save_instead=False):
    imgs = (imgs*255).astype(np.uint8)
    rows = math.ceil(math.sqrt(len(imgs)))
    cols = np.ceil(len(imgs) / float(rows))
    #plt.imshow(imgs[0, :, :, :])
    #plt.show()
    for i in range(len(imgs)):
        plt.subplot(cols, rows, i+1)
        plt.imshow(imgs[i, :, :, :])
        plt.axis('off')
        #plt.tight_layout()
    plt.title(title)
    if not save_instead:
        plt.show()
    else:
        if not os.path.isdir('logs'):
            os.makedirs('logs')
        fname = 'logs/' + title + str(datetime.datetime.now()) + '.jpg'
        plt.savefig(fname)
