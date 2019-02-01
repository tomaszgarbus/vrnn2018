from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
import numpy as np
from tools.visualise import show_images, ChartGenerator
import matplotlib.pyplot as plt
import os.path
from random import shuffle
import sys
from tools.visualise import show_images

SIZE = 64
D_G_RATIO = 2


class DCGans:
    def __init__(self,
                 verbosity=2,
                 input_space_size=200,
                 filename_pref="dc_model"
                 ):
        self.verbosity = verbosity
        self.filename = filename_pref
        self.input_space_size = input_space_size
        if not self.load_if_possible():
            self.generator = self.build_generator()
            self.discriminator = self.build_discriminator()
            self.gan = self.build_gan()
        pass

    def save(self):
        self.gan.save(self.filename + ".h5")

    # TODO przetestowac czy dziala, nie mozna oddzielnie załadować bo musza wskazywać na czesci tego samego obiektu
    def load_if_possible(self):
        filepath = self.filename + ".h5"
        if not os.path.isfile(filepath):
            return False
        self.gan = load_model(filepath)
        print("Ładowanie może nie działać, patrz na kod dcgan_model.py/load_if_possible")
        self.generator = self.gan.layers[1]
        self.discriminator = self.gan.layers[2]
        self.compile_networks()
        return True

    def build_generator(self):
        model = Sequential()
        model.add(Dense((SIZE // 8) ** 2 * 512, use_bias=False, input_shape=(self.input_space_size,)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Reshape((SIZE // 8, SIZE // 8, 512)))
        assert model.output_shape == (None, SIZE // 8, SIZE // 8, 512)

        model.add(Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, SIZE // 4, SIZE // 4, 256)
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, SIZE // 2, SIZE // 2, 128)
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                  activation='sigmoid'))
        assert model.output_shape == (None, SIZE, SIZE, 3)

        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=(SIZE, SIZE, 3)))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(MaxPooling2D((2, 2)))  # 32

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(MaxPooling2D((2, 2)))  # 16

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(MaxPooling2D((2, 2)))  # 8

        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(MaxPooling2D((2, 2)))  # 4

        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        return model

    def compile_networks(self):
        self.discriminator.trainable = False
        self.generator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, beta_1=0.5))
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, beta_1=0.5), metrics=['accuracy'])

    def build_gan(self):
        self.compile_networks()

        z = Input(shape=(self.input_space_size,))
        img = self.generator(z)
        combined = self.discriminator(img)
        gan = Model(z, combined)
        self.discriminator.trainable = False
        gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, beta_1=0.5))
        self.discriminator.trainable = True
        if self.verbosity > 1:
            self.generator.summary()
            self.discriminator.summary()
            gan.summary()
        return gan

    def fit(self, dataset, n_epoch=10, batch_size=16):

        batch_count = dataset.shape[0] // batch_size
        chart = ChartGenerator()

        for i in range(n_epoch):
            lds = []
            lgs = []
            daccs = []
            for j in tqdm(range(batch_count), desc="Epoch " + str(i) if self.verbosity > 0 else "", smoothing=0):
                # Input for the generator
                noise_input = (np.random.rand(batch_size, self.input_space_size) - 0.5) * 2

                for k in range(D_G_RATIO):
                    # getting random images from X_train of size=batch_size
                    # these are the real images that will be fed to the discriminator
                    image_batch = dataset[np.random.randint(0, dataset.shape[0], size=batch_size)]
                    # these are the predicted images from the generator
                    predictions = self.generator.predict(noise_input, batch_size=batch_size)
                    # the discriminator takes in the real images and the generated images
                    X = np.concatenate([predictions, image_batch])
                    # labels for the discriminator
                    y_discriminator = [0] * batch_size + [1] * batch_size
                    # Let's train the discriminator
                    ldisc = self.discriminator.train_on_batch(X, y_discriminator)

                # Let's train the generator
                noise_input = (np.random.rand(batch_size, self.input_space_size) - 0.5) * 2
                y_generator = [1] * batch_size
                # for z in range(3):
                lgen = self.gan.train_on_batch(noise_input, y_generator)

                lds.append(ldisc[0])
                daccs.append(ldisc[1])
                lgs.append(lgen)
            chart.log_values((i+1) * dataset.shape[0], {
                'discriminator_loss': np.mean(lds), 'discriminator_acc': np.mean(daccs),
                'generator_loss': np.mean(lgs)
            })
#            chart.show_chart()
            show_images(predictions, title='G', save_instead=True)
            print("Mean Losses: \nDiscriminator: " + str(np.mean(lds)) + ", "
                  + str(np.mean(daccs))+"\nGenerator: " + str(np.mean(lgs)) + "\n")
            sys.stdout.flush()

            self.save()

    def generate_image(self, seed):
        return self.generator.predict(seed)

    def generate_random_images(self, count=1):
        seed = np.random.rand(count, self.input_space_size)
        return self.generate_image(seed)


