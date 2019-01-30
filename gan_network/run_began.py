from gan_network.began_model import Began
from gan_network.preprocess import Preprocess, BACKGROUND_COLOR_RAND
from tools.visualise import show_images
import numpy as np


if __name__ == '__main__':
    preprocess = Preprocess( cache_file="gan_prep_cache_32")
    dataset = preprocess.load_dataset()

    sample_dataset = dataset[np.random.randint(0, dataset.shape[0], size=9)]
    show_images(sample_dataset)

    model = Began(gamma=0.5, input_space_size=100, filters=64, img_size=32)
    model.fit(dataset, batch_size=50, n_epoch=2000)

    sample_gans = model.generate_random_images(9)
    show_images(sample_gans)

