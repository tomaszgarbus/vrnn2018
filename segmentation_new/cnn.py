import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dropout, Conv2DTranspose, Cropping2D
from keras.activations import relu, sigmoid
from keras.optimizers import Adam

from segmentation_new.constants import SAVED_MODEL_PATH, INPUT_SIZE
from segmentation_new.cars_loader import CarsLoader


class FCN32:
    def __init__(self):
        self.conv_filters = 4096
        self.dropout = 0.5
        self.optimizer = Adam
        self.lr = 1e-4
        self.loss = 'binary_crossentropy'


        self.model = self.load()
        if self.model is None:
            self.model = self.create()

    def create(self):
        print("Creating new model...")

        nb_classes = 2

        inputs = Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))

        x = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)

        x = Conv2D(self.conv_filters, (7, 7), padding='same', activation=relu)(x.output)
        x = Dropout(self.dropout)(x)
        x = Conv2D(self.conv_filters, (1, 1), padding='same', activation=relu)(x)
        x = Dropout(self.dropout)(x)
        x = Conv2D(nb_classes, (1, 1))(x)
        x = Conv2DTranspose(nb_classes, (64, 64), strides=(32, 32), padding='same', activation=sigmoid)(x)
        # x = Cropping2D(cropping=((1, 2), (0, 0)))(x)

        model = Model(inputs=inputs, outputs=x)

        for layer in model.layers[:16]:
            layer.trainable = False

        model.compile(loss=self.loss, optimizer=self.optimizer(lr=self.lr), metrics=['acc'])
        print("Model compiled!")
        return model


    def load(self):
        print("Loading the model from " + SAVED_MODEL_PATH + " ...")
        try:
            model = load_model(filepath=SAVED_MODEL_PATH)
            print("Model loaded!")
        except:
            print("Model not found!")
            model = None
        return model


    def fit(self, train_x, train_y, val_x, val_y, mb_size=2, nb_epochs=1):
        self.model.fit(train_x, train_y, epochs=nb_epochs, batch_size=mb_size, shuffle=True, validation_data=(val_x, val_y))


    def predict(self, xs):
        """
        Returns predictions as binary array.
        """
        onehots = self.model.predict(xs)
        preds = []
        for i in range(onehots.shape[0]):
            preds.append(CarsLoader.onehot_to_binary(onehots[i]))
        return np.array(preds)


    def save(self):
        print("Saving model to " + SAVED_MODEL_PATH + "...")
        self.model.save(SAVED_MODEL_PATH)
        print("Model saved!")


    def summary(self):
        self.model.summary()


if __name__ == '__main__':
    net = FCN32()
    net.save()
