import os
import sys

from rich import print
from tensorflow.python.keras.metrics import accuracy

from nn.wandb_classification_callback import WandbClassificationCallback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import datetime
import random
import time
from collections import Counter
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb

from data_analysis.py_color import PyColor
from data_analysis.utils import Utils
from nn.losses import EDLLoss
from nn.model_base import VDANN

# from nn.metrics import CategoricalTruePositives
from pre_process.pre_process import PreProcess
from pre_process.record import Record

# from wandb.keras import WandbCallback


def generate_and_save_images(
    model,
    epoch,
    test_sample,
):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap="gray")
        plt.axis("off")

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig("image_at_epoch_{:04d}.png".format(epoch))
    plt.show()


def set_seed(seed=200):
    tf.random.set_seed(seed)
    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.0
    return np.where(images > 0.5, 1.0, 0.0).astype("float32")


def main(
    epochs: int = 1,
    batch_size: int = 32,
    has_attention: bool = False,
    has_inception: bool = True,
):

    # データセットの作成
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)
    train_size = 60000
    batch_size = 32
    test_size = 10000
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(x_train)
        .shuffle(train_size)
        .batch(batch_size)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices(x_test)
        .shuffle(test_size)
        .batch(batch_size)
    )
    data_type = "spectrogram"
    shape = (28, 28, 1)
    inputs = tf.keras.Input(shape=shape)

    epochs = 10
    # set the dimensionality of the latent space to a plane for visualization later
    latent_dim = 2
    model = VDANN(
        inputs=inputs,
        gamma=1,
        latent_dim=6,
        alpha=0,
        beta=0,
        target_dim=5,
        subject_dim=68,
        has_inception=has_inception,
        has_attention=has_attention,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
    )
    # Pick a sample of the test set for generating output images
    num_examples_to_generate = 16
    assert batch_size >= num_examples_to_generate
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :, :]

    # keeping the random vector constant for generation (prediction) so
    # it will be easier to see the improvement.
    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim]
    )

    # model_baes内にGPUの計算中にnumpyに渡すことが出来ないのでmainにlambda式用意
    tensor2numpy = lambda x: x.numpy()
    for epoch in range(epochs):
        test_loss_metric = tf.keras.metrics.Mean()
        # print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(train_dataset):
            train_loss = model.train_step(x_batch_train)
            # if step % 50 == 0:
            #     print(
            #         f"train loss: (vae, sbj, tar) =  {tuple(map(tensor2numpy, train_loss))}"
            #     )
        # train_metrics.reset_states()

        for step, x_batch_test in enumerate(test_dataset):
            test_loss = model.test_step(x_batch_test)
            test_loss_metric(test_loss)
        elbo = -test_loss_metric.result()
        # print(
        #     f"test loss: (vae, sbj, tar) = {tuple(map(tensor2numpy, test_loss))}"
        # )
        print(f"Epoch: {epoch}, Test set ELBO: {elbo}")
        # test_metrics.reset_states()


if __name__ == "__main__":
    # シードの固定
    set_seed()
    # 環境設定
    CALC_DEVICE = "gpu"
    DEVICE_ID = "0" if CALC_DEVICE == "gpu" else "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID
    if os.environ["CUDA_VISIBLE_DEVICES"] != "-1":
        tf.keras.backend.set_floatx("float32")
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # tf.config.run_functions_eagerly(True)
    else:
        print("*** cpuで計算します ***")
        # なんか下のやつ使えなくなっている、、
        tf.config.run_functions_eagerly(True)

    # ハイパーパラメータの設定
    TEST_RUN = False
    EPOCHS = 50
    HAS_ATTENTION = False
    PSE_DATA = False
    HAS_INCEPTION = True
    IS_PREVIOUS = False
    IS_NORMAL = True
    HAS_DROPOUT = True
    IS_ENN = False
    # FIXME: 多層化はとりあえずいらない
    IS_MUL_LAYER = True
    HAS_NREM2_BIAS = False
    HAS_REM_BIAS = False
    DROPOUT_RATE = 0.2
    BATCH_SIZE = 64
    N_CLASS = 5
    # KERNEL_SIZE = 512
    # KERNEL_SIZE = 256
    KERNEL_SIZE = 128
    STRIDE = 16
    # STRIDE = 16
    SAMPLE_SIZE = 10000
    DATA_TYPE = "spectrogram"
    FIT_POS = "middle"
    CLEANSING_TYPE = "no_cleansing"
    NORMAL_TAG = "normal" if IS_NORMAL else "sas"
    ATTENTION_TAG = "attention" if HAS_ATTENTION else "no-attention"
    PSE_DATA_TAG = "psedata" if PSE_DATA else "sleepdata"
    INCEPTION_TAG = "inception" if HAS_INCEPTION else "no-inception"
    WANDB_PROJECT = "test" if TEST_RUN else "1215_test"
    # WANDB_PROJECT = "test" if TEST_RUN else "base_learning_20211109"
    ENN_TAG = "enn" if IS_ENN else "dnn"
    INCEPTION_TAG += "v2" if IS_MUL_LAYER else ""

    main(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        has_attention=HAS_ATTENTION,
        has_inception=HAS_INCEPTION,
    )
