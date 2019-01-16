from gan_network.began_model import Began
from gan_network.preprocess import Preprocess
from tools.visualise import show_images
import numpy as np


if __name__ == '__main__':
    preprocess = Preprocess()
    dataset = preprocess.load_dataset()

    sample_dataset = dataset[np.random.randint(0, dataset.shape[0], size=9)]
    show_images(sample_dataset)

    model = Began(gamma=0.7, filters=64)
    model.fit(dataset, batch_size=4)

    sample_gans = model.generate_random_images(9)
    show_images(sample_gans)
