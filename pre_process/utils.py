import os
import random

import numpy as np
import tensorflow as tf


class Utils:
    def __init__(self) -> None:
        pass


# MNISTデータの前処理
def preprocess_images(images: tf.Tensor):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.0
    return np.where(images > 0.5, 1.0, 0.0).astype("float32")


def set_seed(seed=200):
    tf.random.set_seed(seed)
    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    return
