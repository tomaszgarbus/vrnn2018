from gan_network.began_model import Began
from gan_network.preprocess import Preprocess
from tools.visualise import show_images
import numpy as np


if __name__ == '__main__':
    preprocess = Preprocess(x_size=64, y_size=64, cache_file="gan_cache_64")
    dataset = preprocess.load_dataset()

    org_imgs = dataset[np.random.randint(0, dataset.shape[0], size=2)]
    fst = org_imgs[0, :, :, :].reshape((1, org_imgs.shape[1], org_imgs.shape[2], org_imgs.shape[3]))
    lst = org_imgs[1, :, :, :].reshape((1, org_imgs.shape[1], org_imgs.shape[2], org_imgs.shape[3]))
    show_images(org_imgs, title="original images")
    img_number = 9

    model = Began(gamma=0.3, filters=64, img_size=64, filename_pref="model_64")
    code1 = model.find_code_for_image(fst)
    code2 = model.find_code_for_image(lst)

    code_step = (code2 - code1)/(img_number - 1)
    codes = [code1 + code_step*i for i in range(img_number)]
    imgs = [model.generate_image(code) for code in codes]
    show_images(np.concatenate(imgs), title="image transition")

