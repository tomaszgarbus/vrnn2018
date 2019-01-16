import numpy as np
import logging
from math import ceil
from progress.bar import Bar
from segmentation_network.constants import INPUT_SIZE
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, Conv2DTranspose, Cropping2D
from keras.optimizers import Adam

SAVED_MODEL_PATH = 'vgg_segmentation_tmp'


class VGGNet:
    def __init__(self):
        inputs = Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))
        vgg = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
        x = Conv2D(filters=2048, kernel_size=(7, 7), activation='relu', padding='same')(vgg.output)
        x = Dropout(0.5)(x)
        x = Conv2D(filters=2048, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = Dropout(0.5)(x)
        x = Conv2D(filters=1, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = Conv2DTranspose(filters=1, kernel_size=(64, 64), strides=(33, 32), padding='same',
                            activation='sigmoid')(x)
        x = Cropping2D(cropping=((3, 0), (0, 0)))(x)
        self.model = Model(inputs=inputs, outputs=x)
        # ... disable training of first e.g. 15 layers
        for layer in vgg.layers[:17]:
            #   print(layer, layer in vgg.layers)
            layer.trainable = False

        self.model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        pass

    def fit(self, x, y, mb_size=2, nb_epochs=1):
        for epoch in range(nb_epochs):
            print("Epoch {0}/{1}".format(epoch + 1, nb_epochs))
            logging.info("Epoch {0}/{1}".format(epoch + 1, nb_epochs))
            iters = int(ceil(len(x)/mb_size))
            bar = Bar(max=iters)
            sum_accs = 0.
            for iter in range(iters):
                # Shuffle the training set.
                order = list(range(x.shape[0]))
                np.random.shuffle(order)
                x = x[order]
                y = y[order]

                batch_x = x[iter * mb_size: (iter + 1) * mb_size]
                batch_y = y[iter * mb_size: (iter + 1) * mb_size]
                loss, acc = self.model.train_on_batch(batch_x, batch_y)

                sum_accs += acc
                bar.message = 'loss: {0:.8f} acc: {1:.4f} mean_acc: {2:.4f}'.format(loss, acc, sum_accs/(iter+1))
                bar.next()
            bar.finish()
            self.save()

    def predict(self, x):
        return self.model.predict(x)

    def save(self):
        self.model.save(SAVED_MODEL_PATH)