from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


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
        self.gan = load_model(self.filename + ".h5")
        print("Ładowanie może nie działać, patrz na kod dcgan_model.py/load_if_possible")
        self.generator = self.gan.layers[1]
        self.discriminator = self.gan.layers[2]

    def build_generator(self):
        model = Sequential()
        model.add(Dense(8 * 8 * 320 * 3, use_bias=False, input_shape=(self.input_space_size,)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Reshape((8, 8, 320 * 3)))
        assert model.output_shape == (None, 8, 8, 320 * 3)

        model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 8, 8, 192 * 3)
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 16, 16, 128 * 3)
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 32, 32, 64 * 3)
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 64, 64, 3)

        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=(64, 64, 3)))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(MaxPooling2D((2, 2)))  # 32

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(MaxPooling2D((2, 2)))  # 16

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(MaxPooling2D((2, 2)))  # 8

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(MaxPooling2D((2, 2)))  # 4

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(MaxPooling2D((2, 2)))  # 2

        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        return model

    def build_gan(self):
        self.generator.compile(loss='binary_crossentropy', optimizer='adam')
        self.discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        z = Input(shape=(self.input_space_size,))
        img = self.generator(z)
        combined = self.discriminator(img)
        gan = Model(z, combined)
        gan.compile(loss='binary_crossentropy', optimizer='adam')
        if self.verbosity > 1:
            self.generator.summary()
            self.discriminator.summary()
            gan.summary()
        return gan

    def fit(self, dataset, n_epoch=1, batch_size=16):

        batch_count = dataset.shape[0] // batch_size

        for i in range(n_epoch):
            for j in tqdm(range(batch_count), desc="Epoch " + str(n_epoch) if self.verbosity > 0 else ""):
                # Input for the generator
                noise_input = np.random.rand(batch_size, self.input_space_size)

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
                self.discriminator.trainable = True
                self.discriminator.train_on_batch(X, y_discriminator)

                # Let's train the generator
                noise_input = np.random.rand(batch_size, self.input_space_size)
                y_generator = [1] * batch_size
                self.discriminator.trainable = False
                self.gan.train_on_batch(noise_input, y_generator)

    def generate_image(self, seed):
        return self.generator.predict(seed)

    def generate_random_images(self, count=1):
        seed = np.random.rand(self.input_space_size, count)
        return self.generate_image(seed)

    def show_gen_images(self, count=9):
        plt.figure(figsize=(64, 64))
        for i in range(preds.shape[0]):
            plt.subplot(64, 64, i + 1)
            plt.imshow(preds[i, :, :, 0])
            plt.axis('off')
            plt.tight_layout()



