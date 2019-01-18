from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam, SGD
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os.path
from tools.visualise import show_images
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras import backend as bkeras


class Began:
    def __init__(self,
                 verbosity=2,
                 input_space_size=200,
                 filename_pref="be_model",
                 img_size=128,
                 filters=128,
                 gamma=0.3,
                 log_stats_per=1,
                 log_image_per=20
                 ):
        self.img_size = img_size
        self.log_stats_per = log_stats_per
        self.log_image_per = log_image_per
        self.verbosity = verbosity
        self.filename = filename_pref
        self.input_space_size = input_space_size
        self.filters = filters
        self.adam = Adam(lr=0.00005)  # lr: between 0.0001 and 0.00005
        self.adam_gen = Adam(lr=0.00005)

        if not self.load_if_possible():
            self.generator = self.build_decoder()
            self.discriminator = self.build_autoencoder()
            self.gan = self.build_gan()

        if self.verbosity > 1:
            self.generator.summary()
            self.discriminator.summary()
            self.gan.summary()

        self.epsilon = K.epsilon()
        self.k = self.epsilon
        self.kLambda = 0.0003
        self.gamma = gamma

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
        self.generator.compile(loss='mean_absolute_error', optimizer=self.adam_gen)
        self.discriminator.trainable = True
        self.discriminator.compile(loss='mean_absolute_error', optimizer=self.adam)

    def build_gan(self):
        self.compile_networks()
        z = Input(shape=(self.input_space_size,))
        img = self.generator(z)
        self.discriminator.trainable = False
        combined = self.discriminator(img)
        gan = Model(z, combined)
        gan.compile(loss='mean_absolute_error', optimizer=self.adam)
        self.discriminator.trainable = True
        return gan

    def find_code_for_image(self, img, iterations=1000, log_for=100, loss_eps = 0.01):
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
                print("current loss: " + str(loss))
        show_lay = Model(dummy_input_layer, input_to_fit)
        show_lay.compile(loss='mean_absolute_error', optimizer='SGD')  # dummy compile, only to call predict
        optimized_code = show_lay.predict(dummy_input)
        self.generator.trainable = True
        return optimized_code

    def fit(self, dataset, n_epoch=10, batch_size=16):

        batch_count = dataset.shape[0] // batch_size

        lds = []
        lgs = []

        for i in range(n_epoch):
            trange = tqdm(range(batch_count), desc="Epoch " + str(i) if self.verbosity > 0 else "", smoothing=0)
            for j in trange:

                # Let's train the discriminator
                # On generated images
                noise_input_disc = np.random.uniform(-1, 1, (batch_size, self.input_space_size))
                x_gen = self.generator.predict(noise_input_disc)
                d_loss_gen = self.discriminator.train_on_batch(x_gen, x_gen, -self.k * np.ones(batch_size))

                # On real images
                image_batch = dataset[np.random.randint(0, dataset.shape[0], size=batch_size)]
                x_real = image_batch
                d_loss_real = self.discriminator.train_on_batch(x_real, x_real)

                d_loss = d_loss_real + d_loss_gen

                # Let's train the generator
                noise_input_gen = np.random.uniform(-1, 1, (batch_size * 2, self.input_space_size))
                predictions = self.gan.predict(noise_input_gen, batch_size=batch_size)
                gen_loss = self.generator.train_on_batch(noise_input_gen, predictions)

                # Now update k
                old_k = self.k
                self.k = self.k + self.kLambda * (self.gamma * d_loss_real - d_loss_gen)
                self.k = min(max(self.k, self.epsilon), 1)

                # Calculate the global measure
                m_global = d_loss_real + np.abs(self.gamma * d_loss_real - d_loss_gen)
                gamma_real = d_loss_gen / d_loss_real

                if j % self.log_image_per == 0:
                    d_real_predictions = self.discriminator.predict(x_real)
                    d_gen_predictions = self.discriminator.predict(x_gen)
                    show_images(d_real_predictions, title="D_real", save_instead=False)
                    show_images(d_gen_predictions, title="D_gen", save_instead=False)
                    show_images(predictions, title="G", save_instead=False)

                if j % self.log_stats_per == 0:
                    print("Global measure: " + str(m_global) + " gamma_real: " + str(gamma_real) + " d_loss: " + str(d_loss) + " gen_loss: " + str(gen_loss) +
                              " d_loss_real: " + str(d_loss_real) + " d_loss_gen: " + str((-1/old_k) * d_loss_gen) +
                          " k: " + str(self.k) + "\n")

            self.save()

    # def train(self, nb_epoch, nb_batch_per_epoch, batch_size, gamma, path=""):
    #     '''
    #     Train a Generator network and Discriminator Method using the BEGAN method. The networks are updated sequentially unlike what's done in the paper.
    #     Keyword Arguments:
    #     nb_epoch -- Number of training epochs
    #     batch_size -- Size of a single batch of real data.
    #     nb_batch_per_epoch -- Number of training batches to run each epoch.
    #     gamma -- Hyperparameter from BEGAN paper to regulate proportion of Generator Error over Discriminator Error. Defined from 0 to 1.
    #     path -- Optional parameter specifying location to save output file locations. Starts from the working directory.
    #     '''
    #     for e in range(self.firstEpoch, self.firstEpoch + nb_epoch):
    #         progbar = generic_utils.Progbar(nb_batch_per_epoch * batch_size)
    #         start = time.time()
    #
    #         for b in range(nb_batch_per_epoch):
    #             zD = np.random.uniform(-1, 1, (batch_size, self.z))
    #             zG = np.random.uniform(-1, 1, (batch_size * 2, self.z))  #
    #
    #             # Train D
    #             real = self.dataGenerator.next()
    #             d_loss_real = self.discriminator.train_on_batch(real, real)
    #
    #             gen = self.generator.predict(zD)
    #             weights = -self.k * np.ones(batch_size)
    #             d_loss_gen = self.discriminator.train_on_batch(gen, gen, weights)
    #
    #             d_loss = d_loss_real + d_loss_gen
    #
    #             # Train G
    #             self.discriminator.trainable = False
    #             target = self.generator.predict(zG)
    #             g_loss = self.gan.train_on_batch(zG, target)
    #             self.discriminator.trainable = True
    #
    #             # Update k
    #             self.k = self.k + self.kLambda * (gamma * d_loss_real - g_loss)
    #             self.k = min(max(self.k, self.epsilon), 1)
    #
    #             # Report Results
    #             m_global = d_loss + np.abs(gamma * d_loss_real - g_loss)
    #             progbar.add(batch_size,
    #                         values=[("M", m_global), ("Loss_D", d_loss), ("Loss_G", g_loss), ("k", self.k)])
    #
    #             if (self.logEpochOutput and b == 0):
    #                 with open(os.getcwd() + path + '/output.txt', 'a') as f:
    #                     f.write("{}, M: {}, Loss_D: {}, LossG: {}, k: {}\n".format(e, m_global, d_loss, g_loss,
    #                                                                                self.k))
    #
    #             if (self.sampleSwatch and b % (nb_batch_per_epoch / 2) == 0):
    #                 if (self.saveSampleSwatch):
    #                     genName = '/generatorSample_{}_{}.png'.format(e, int(b / nb_batch_per_epoch / 2))
    #                     discName = '/discriminatorSample_{}_{}.png'.format(e, int(b / nb_batch_per_epoch / 2))
    #                 else:
    #                     genName = '/currentGeneratorSample.png'
    #                     discName = '/currentDiscriminatorSample.png'
    #                 utils.plotGeneratedBatch(real, gen, path + genName)
    #                 utils.plotGeneratedBatch(self.discriminator.predict(real), target, path + discName)
    #
    #         print('\nEpoch {}/{}, Time: {}'.format(e + 1, nb_epoch, time.time() - start))

    def generate_image(self, seed):
        return self.generator.predict(seed)

    def generate_random_images(self, count=1):
        seed = np.random.rand(count, self.input_space_size)
        return self.generate_image(seed)


