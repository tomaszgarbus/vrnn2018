import tensorflow as tf
from typing import Optional, List, Tuple
from math import sqrt, ceil
import logging
from progress.bar import Bar
import matplotlib.pyplot as plt
import numpy as np

from segmentation_network.constants import INPUT_SIZE, DOWNCONV_FILTERS, UPCONV_FILTERS, SAVED_MODEL_PATH, POOL_SIZE

# A single layer filters description. Consists of:
# * number of filters (int)
# * convolution window ([int, int])
# * how many times to repeat the layer (int)
FilterDesc = Tuple[int, List[int], int]


class UNet:
    def _init_defaults(self):
        self.learning_rate = 0.5
        self.nb_iters = 100000
        self.input_size = INPUT_SIZE
        self.downconv_filters = DOWNCONV_FILTERS
        self.upconv_filters = UPCONV_FILTERS

        self.downconv_layers = []
        self.downpool_layers = []
        self.upconv_layers = []

    def __init__(self,
                 sess,
                 learning_rate: Optional[float] = None,
                 nb_epochs: Optional[int] = None,
                 input_size: Optional[List[int]] = None,
                 downconv_filters: Optional[List[FilterDesc]] = None,
                 upconv_filters: Optional[List[FilterDesc]] = None):
        self._init_defaults()

        self.sess = sess
        if learning_rate:
            self.learning_rate = learning_rate
        if nb_epochs:
            self.nb_iters = nb_epochs
        if input_size:
            self.input_size = input_size
        if downconv_filters:
            self.downconv_filters = downconv_filters
        if upconv_filters:
            self.upconv_filters = upconv_filters

        # Initialize logging.
        self.logger = logging.Logger("main_logger", level=logging.INFO)
        formatter = logging.Formatter(
            fmt='{message}',
            style='{'
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        # file_handler = logging.FileHandler(log_file)
        # file_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        # self.logger.addHandler(file_handler)

        self._create_model()

    def _add_downconv_layers(self):
        signal = self.x
        # Initial normalization.
        mean, variance = tf.nn.moments(signal, axes=[0, 1, 2, 3])
        signal = (signal - mean) / tf.sqrt(variance)

        for layer_no in range(len(self.downconv_filters)):
            filters_count, kernel_size, layers_repeat = self.downconv_filters[layer_no]
            for count in range(layers_repeat):
                # Weights initialization (std. dev = sqrt(2 / N))
                cur_shape = tuple(map(int, signal.get_shape()[1:]))
                inputs = cur_shape[0] * cur_shape[1] * cur_shape[2]
                w_init = tf.initializers.random_normal(stddev=sqrt(2 / inputs))
                # Convolutional layer.
                downconv_layer = tf.layers.conv2d(signal,
                                                  filters=filters_count,
                                                  kernel_size=kernel_size,
                                                  padding='same',
                                                  activation=tf.nn.relu,
                                                  kernel_initializer=w_init,
                                                  use_bias=False)  # Bias not needed with batch normalization
                # Batch normalization layer
                batch_norm = tf.layers.batch_normalization(downconv_layer)
                signal = batch_norm
            self.downconv_layers.append(downconv_layer)
            # Downpooling layer.
            downpooling_layer = tf.layers.max_pooling2d(signal,
                                                        pool_size=[POOL_SIZE, POOL_SIZE],
                                                        strides=[POOL_SIZE, POOL_SIZE],
                                                        padding='same')
            self.downpool_layers.append(downpooling_layer)
            signal = downpooling_layer

    def _add_upconv_layers(self):
        signal = self.downpool_layers[-1]
        for layer_no in range(len(self.upconv_filters)):
            filters_count, kernel_size, layer_repeat = self.upconv_filters[layer_no]
            for count in range(layer_repeat):
                # Weights initialization (std. dev = sqrt(2 / N)).
                cur_shape = tuple(map(int, signal.get_shape()[1:]))
                inputs = cur_shape[0] * cur_shape[1] * cur_shape[2]
                w_init = tf.initializers.random_normal(stddev=sqrt(2 / inputs))
                # Convolutional layer.
                upconv_layer = tf.layers.conv2d(signal,
                                                filters=filters_count,
                                                kernel_size=kernel_size,
                                                padding='same',
                                                activation=tf.nn.relu,
                                                kernel_initializer=w_init,
                                                use_bias=False)
                # Batch normalization layer
                batch_norm = tf.layers.batch_normalization(upconv_layer)
                signal = batch_norm
            self.upconv_layers.append(upconv_layer)
            # Concatenates with respective downconv.
            if layer_no and layer_no < len(self.downconv_layers):
                upconv_concat = tf.concat([batch_norm, self.downconv_layers[-layer_no]], axis=3)
                upconv_layer = upconv_concat
            else:
                upconv_layer = batch_norm
            # Upsampling layer.
            if layer_no < len(self.downconv_layers):
                cur_shape = tuple(map(int, upconv_layer.get_shape()[1:]))
                new_shape = (-1, cur_shape[0] * POOL_SIZE, cur_shape[1] * POOL_SIZE, cur_shape[2])
                self.logger.info((cur_shape, new_shape))
                uppooling_layer = tf.image.resize_nearest_neighbor(images=upconv_layer,
                                                                   size=new_shape[1:3])
                signal = uppooling_layer
            else:
                signal = upconv_layer

        self.output = signal
        self.logger.info(signal.get_shape())

    def _add_training_objectives(self):
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.y, self.output))
        self.preds = tf.round(self.output)
        self.labels = self.y
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.preds, self.labels), tf.float32))
        self.train_op = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9).minimize(self.loss)

    def _create_model(self):
        self.x = tf.placeholder(tf.float32, [None] + self.input_size + [3])
        self.y = tf.placeholder(tf.float32, [None] + self.input_size + [1])

        self._add_downconv_layers()
        self._add_upconv_layers()

        self._add_training_objectives()

        # Restores model if possible.
        self.saver = tf.train.Saver()
        try:
            self.saver.restore(self.sess, SAVED_MODEL_PATH)
        except:
            # Initializes variables.
            print("Initializing variables (model checkpoint not found)")
            tf.global_variables_initializer().run()

    def fit(self, x, y, mb_size=2, nb_epochs=1, validation_data=None):
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
                loss, acc, _, output, gt = self.sess.run([self.loss,
                                                          self.accuracy,
                                                          self.train_op,
                                                          self.preds,
                                                          self.labels],
                                                         feed_dict={
                                                            self.x: batch_x,
                                                            self.y: batch_y
                                                         })
                sum_accs += acc
                bar.message = 'loss: {0:.8f} acc: {1:.4f} mean_acc: {2:.4f}'.format(loss, acc, sum_accs/(iter+1))
                bar.next()
            bar.finish()
            if not (validation_data is None):
                val_acc = self.evaluate(validation_data[0], validation_data[1])
                print('val_acc: {0:.4f}'.format(val_acc))
            self.save()

    def predict(self, x):
        results = self.sess.run([self.preds], feed_dict={self.x: x})
        return results[0]

    def evaluate(self, x: np.ndarray, y: np.ndarray, mb_size=2):
        assert len(x) == len(y)
        iters = int(ceil(len(x)/mb_size))
        preds = []
        for iter in range(iters):
            batch_x = x[iter * mb_size: (iter + 1) * mb_size]
            preds.append(self.predict(batch_x))
        pred = np.concatenate(preds, axis=0)
        pred.reshape(y.shape)
        corr = (pred == y).sum()
        acc = corr / pred.size
        return acc

    def save(self):
        self.saver.save(self.sess, SAVED_MODEL_PATH)


if __name__ == '__main__':
    with tf.Session() as sess:
        net = UNet(sess,
                   learning_rate=0.1)
        net.save()
    pass
