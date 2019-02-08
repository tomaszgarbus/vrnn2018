from gan_network.dcgan_model import  DCGans
from gan_network.preprocess import Preprocess
from tools.visualise import show_images
import numpy as np
import gan_network.inception_score as IS


if __name__ == '__main__':
    preprocess = Preprocess()
    dataset = preprocess.load_dataset() * 255

    sample_dataset = dataset[np.random.randint(0, dataset.shape[0], size=9)]
    show_images(sample_dataset)

    dataset_imgs = []
    for i in range(dataset.shape[0]):
        dataset_imgs.append(dataset[i])
    print(IS.get_inception_score(dataset_imgs))
    exit(0)

    model = DCGans()
    model.fit(dataset, n_epoch=100, batch_size=64)

    sample_gans = model.generate_random_images(9)
    show_images(sample_gans)

