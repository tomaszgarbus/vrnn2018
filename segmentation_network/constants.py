# Single image input size to the neural net.
INPUT_SIZE = [256, 256]

# Describes the downscaling convolution filters.
# Required format: List[Tuple[Int, Tuple[Int, Int], Int]]
# (number of filters, (convolution window), repetitions of this layer).
DOWNCONV_FILTERS = [(20, [5, 5], 1), (64, [5, 5], 1)]

# Describes the upscaling convolution filters.
UPCONV_FILTERS = [(64, [5, 5], 1), (64, [5, 5], 1), (1, [5, 5], 1)]

# Path to the saved model.
SAVED_MODEL_PATH = 'segmentation_tmp/model.ckpt'

# Pool size for max-pooling.
POOL_SIZE = 4
