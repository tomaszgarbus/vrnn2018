from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
import numpy as np
from tools.visualise import show_images, ChartGenerator
import os.path
import sys
from tools.visualise import show_images

SIZE = 64
D_ITER = 1
G_ITER = 1


class DCGans:
    def __init__(self,
                 verbosity=2,
                 input_space_size=200,
                 filename_pref="dc_model"
                 ):
        self.verbosity = verbosity
        self.gan_filename = filename_pref
        self.input_space_size = input_space_size
        if not self.load_if_possible():
            self.generator = self.build_generator()
            self.discriminator = self.build_discriminator()
            self.gan = self.build_gan()
        pass

    def save(self, iter_no=None):
        if iter_no is not None:
            self.gan.save(self.gan_filename + str(iter_no) + ".h5")
        else:
            self.gan.save(self.gan_filename + ".h5")

    # TODO przetestowac czy dziala, nie mozna oddzielnie załadować bo musza wskazywać na czesci tego samego obiektu
    def load_if_possible(self):
        # TODO: generator i dyskryminator też
        filepath = self.gan_filename + ".h5"
        if not os.path.isfile(filepath):
            return False
        self.gan = load_model(filepath)
        self.discriminator = load_model(filepath)
        print("Ładowanie może nie działać, patrz na kod dcgan_model.py/load_if_possible")
        self.generator = self.gan.layers[1]
        self.discriminator = self.gan.layers[2]
        self.compile_networks()
        return True

    def build_generator(self):
        model = Sequential()
        model.add(Dense((SIZE // 4) ** 2 * 512, use_bias=False, input_shape=(self.input_space_size,)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Reshape((SIZE // 4, SIZE // 4, 512)))
        assert model.output_shape == (None, SIZE // 4, SIZE // 4, 512)

        # model.add(Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        # assert model.output_shape == (None, SIZE // 4, SIZE // 4, 512)
        # model.add(BatchNormalization())
        # model.add(LeakyReLU())

        model.add(Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, SIZE // 2, SIZE // 2, 256)
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

        # model.add(Conv2D(512, (3, 3), padding='same'))
        # model.add(LeakyReLU())
        # model.add(Dropout(0.5))
        # model.add(MaxPooling2D((2, 2)))  # 4

        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        return model

    def compile_networks(self):
        self.discriminator.trainable = False
        self.generator.compile(loss='binary_crossentropy', optimizer=Adam(0.002, beta_1=0.5))
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.002, beta_1=0.5), metrics=['accuracy'])

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

                for k in range(D_ITER):
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

                for k in range(G_ITER):
                    # Let's train the generator
                    noise_input = (np.random.rand(batch_size, self.input_space_size) - 0.5) * 2
                    y_generator = [1] * batch_size
                    # for z in range(3):
                    lgen = self.gan.train_on_batch(noise_input, y_generator)

                lds.append(ldisc[0])
                daccs.append(ldisc[1])
                lgs.append(lgen)
                if j%50 == 0:
                    print("Mean Losses: \nDiscriminator: " + str(np.mean(lds)) + ", "
                          + str(np.mean(daccs)) + "\nGenerator: " + str(np.mean(lgs)) + "\n")
            chart.log_values(batch_count * (i+1), {
                'd_loss': np.mean(lds), 'gen_loss': np.mean(lgs), 'd_acc': np.mean(daccs),
            })
            chart.show_chart(ylim_max=8, title='dcgan')
            show_images(self.choose_best(predictions), count=16, title='G', save_instead=True)
            print("Mean Losses: \nDiscriminator: " + str(np.mean(lds)) + ", "
                  + str(np.mean(daccs))+"\nGenerator: " + str(np.mean(lgs)) + "\n")
            sys.stdout.flush()

            self.save()
            self.save(i)

    def generate_image(self, seed):
        return self.generator.predict(seed)

    def choose_best(self, images: np.ndarray, count=16):
        """ From generated images chooses the ones that have the highest discriminator loss. """
        dis_preds = self.discriminator.predict(images)
        indices = list(range(len(images)))
        indices.sort(key=lambda idx: dis_preds[idx], reverse=True)
        return images[indices[:count]]

    def generate_random_images(self, count=1):
        seed = np.random.rand(count, self.input_space_size)
        return self.generate_image(seed)

    def find_code_for_image(self, img, iterations=1000, log_for=100, loss_eps=0.01):
        self.generator.trainable = False
        dummy_input = np.random.uniform(-1, 1, (1, self.input_space_size))
        dummy_input_layer = Input(shape=(self.input_space_size,))
        input_to_fit = Dense(self.input_space_size)(dummy_input_layer)
        last = self.generator(input_to_fit)
        optimizer = Model(dummy_input_layer, last)
        optimizer.compile(loss='mean_absolute_error', optimizer='SGD')
        for i in range(iterations):
            loss = optimizer.train_on_batch(dummy_input, img)
            if loss < loss_eps:
                break
            if i % log_for == 0:
                print("iter: " + str(i) + " current loss: " + str(loss))
        show_lay = Model(dummy_input_layer, input_to_fit)
        show_lay.compile(loss='mean_absolute_error', optimizer='SGD')  # dummy compile, only to call predict
        optimized_code = show_lay.predict(dummy_input)
        self.generator.trainable = True
        return optimized_code


