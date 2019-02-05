import matplotlib.pyplot as plt
import numpy as np
import math
import os
import datetime
import sys


def show_images(imgs, title="", save_instead=False):
    imgs = imgs[:9, :, :, :]
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


class ChartGenerator:

    def __init__(self, print_logged=False):
        self.logged = {'iter':[]}
        self.print_logged = print_logged

    def log_values(self, iter_overall, new_vals):
        self.logged['iter'].append(iter_overall)
        for (k, v) in new_vals.items():
            w = self.logged.get(k, [])
            w.append(v)
            self.logged[k] = w
        if self.print_logged:
            message = [k + ": " + str(v) for (k, v) in new_vals.items()]
            print(" ".join(message))
            sys.stdout.flush()

    def show_chart(self):
        plt.style.use('seaborn-darkgrid')
        my_dpi = 96
        plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
        #palette = plt.get_cmap('plasma')
        for (i, column) in enumerate(self.logged.keys()):
            if column != "iter":
                plt.plot('iter', column, data=self.logged, marker='',  linewidth=2, alpha=0.7, label=column)

        plt.legend()

        plt.title("began", loc='left', fontsize=12, fontweight=0, color='orange')
        plt.xlabel("iterations")
        plt.ylabel("value")
        plt.ylim(-0.01, 0.5)
        fname = 'logs/' + "PL_ALL" + str(datetime.datetime.now()) + '.jpg'
        plt.savefig(fname)
        plt.close()
        if 'gamma_real' not in self.logged:
            return
        plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
        plt.plot('iter', 'gamma_real', data=self.logged, color='orange', marker='', linewidth=2, alpha=0.4, label='gamma_real')
        plt.title('gamma real')
        fname = 'logs/' + "PL_GAMMA" + str(datetime.datetime.now()) + '.jpg'
        plt.savefig(fname)
        plt.close()





