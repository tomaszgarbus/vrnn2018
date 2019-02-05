from gan_network.dcgan_model import  DCGans
from gan_network.preprocess import Preprocess
from tools.visualise import show_images
import numpy as np


if __name__ == '__main__':
    preprocess = Preprocess()
    dataset = preprocess.load_dataset()

    sample_dataset = dataset[np.random.randint(0, dataset.shape[0], size=9)]
    show_images(sample_dataset)

    model = DCGans()
    model.fit(dataset, n_epoch=20, batch_size=64)

    sample_gans = model.generate_random_images(9)
    show_images(sample_gans)

