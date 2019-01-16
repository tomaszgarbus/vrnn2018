from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os.path
from tools.visualise import show_images
from keras.layers.convolutional import Convolution2D, UpSampling2D


def l1Loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))


class Began:
    def __init__(self,
                 verbosity=2,
                 input_space_size=200,
                 filename_pref="dc_model",
                 img_size=128,
                 filters=128
                 ):
        self.img_size = img_size
        self.verbosity = verbosity
        self.filename = filename_pref
        self.input_space_size = input_space_size
        self.filters = filters
        if not self.load_if_possible():
            self.generator = self.build_encoder()
            self.discriminator = self.build_autoencoder()
            self.gan = self.build_gan()

        if self.verbosity > 1:
            self.generator.summary()
            self.discriminator.summary()
            self.gan.summary()
        pass

    def save(self):
        self.gan.save(self.filename + ".h5")

    def load_if_possible(self):
        filepath = self.filename + ".h5"
        if not os.path.isfile(filepath):
            return False
        self.gan = load_model(filepath)
        self.generator = self.gan.layers[1]
        self.discriminator = self.gan.layers[2]
        self.compile_networks()
        return True

    def build_encoder(self):
        init_dim = 8
        layers = int(np.log2(self.img_size) - 2)

        mod_input = Input(shape=(self.img_size, self.img_size, 3))
        x = Convolution2D(3, 3, 3, activation='elu', border_mode="same")(mod_input)

        for i in range(1, layers):
            x = Convolution2D(i * self.filters, 3, 3, activation='elu', border_mode="same")(x)
            x = Convolution2D(i * self.filters, 3, 3, activation='elu', border_mode="same", subsample=(2, 2))(x)

        x = Convolution2D(layers * self.filters, 3, 3, activation='elu', border_mode="same")(x)
        x = Convolution2D(layers * self.filters, 3, 3, activation='elu', border_mode="same")(x)

        x = Reshape((layers * self.filters * init_dim ** 2,))(x)
        x = Dense(self.input_space_size)(x)

        return Model(mod_input, x)

    def build_decoder(self):
        init_dim = 8  # Starting size from the paper
        layers = int(np.log2(self.img_size) - 3)

        mod_input = Input(shape=(self.input_space_size,))
        x = Dense(self.filters * init_dim ** 2)(mod_input)
        x = Reshape((init_dim, init_dim, self.filters))(x)

        x = Convolution2D(self.filters, 3, 3, activation='elu', border_mode="same")(x)
        x = Convolution2D(self.filters, 3, 3, activation='elu', border_mode="same")(x)

        for i in range(layers):
            x = UpSampling2D(size=(2, 2))(x)
            x = Convolution2D(self.filters, 3, 3, activation='elu', border_mode="same")(x)
            x = Convolution2D(self.filters, 3, 3, activation='elu', border_mode="same")(x)

        x = Convolution2D(3, 3, 3, activation='elu', border_mode="same")(x)

        return Model(mod_input, x)

    def build_autoencoder(self):

        mod_input = Input(shape=(self.img_size, self.img_size, 3))
        x = self.build_encoder()(mod_input)
        x = self.build_decoder()(x)

        return Model(mod_input, x)

    def compile_networks(self):
        adam = Adam(lr=0.00005)  # lr: between 0.0001 and 0.00005
        self.generator.compile(loss=l1Loss, optimizer=adam)
        self.discriminator.compile(loss=l1Loss, optimizer=adam)
        self.gan.compile(loss=l1Loss, optimizer=adam)

    def build_gan(self):
        self.compile_networks()

        z = Input(shape=(self.input_space_size,))
        img = self.generator(z)
        combined = self.discriminator(img)
        gan = Model(z, combined)
        return gan

    def fit(self, dataset, n_epoch=10, batch_size=16):

        batch_count = dataset.shape[0] // batch_size

        lds = []
        lgs = []

        for i in range(n_epoch):
            for j in tqdm(range(batch_count), desc="Epoch " + str(i) if self.verbosity > 0 else "", smoothing=0):
                # Input for the generator
                noise_input = np.random.rand(batch_size, self.input_space_size)

                # getting random images from X_train of size=batch_size
                # these are the real images that will be fed to the discriminator
                image_batch = dataset[np.random.randint(0, dataset.shape[0], size=batch_size)]

                # these are the predicted images from the generator
                predictions = self.generator.predict(noise_input, batch_size=batch_size)

                if j % 100 == 0:
                    show_images(predictions)

                # the discriminator takes in the real images and the generated images
                X = np.concatenate([predictions, image_batch])

                # labels for the discriminator
                y_discriminator = [0] * batch_size + [1] * batch_size

                # Let's train the discriminator
                self.discriminator.trainable = True
                ldisc = self.discriminator.train_on_batch(X, y_discriminator)

                # Let's train the generator
                noise_input = np.random.rand(batch_size, self.input_space_size)
                y_generator = [1] * batch_size
                self.discriminator.trainable = False
                lgen = self.gan.train_on_batch(noise_input, y_generator)

                lds.append(ldisc)
                lgs.append(lgen)
                if j % 10 == 0:
                    print("Last Losses: \nDiscriminator: " + str(lds[-10:]) + "\nGenerator: " + str(lgs[-10:]) + "\n")

            self.save()

    def generate_image(self, seed):
        return self.generator.predict(seed)

    def generate_random_images(self, count=1):
        seed = np.random.rand(count, self.input_space_size)
        return self.generate_image(seed)


