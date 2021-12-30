import os
import sys

from IPython import display
from rich import print
from tensorflow.python.keras.metrics import accuracy

from nn.wandb_classification_callback import WandbClassificationCallback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import datetime
import random
from collections import Counter
from typing import Any, Dict, List

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import wandb

from data_analysis.py_color import PyColor
from data_analysis.utils import Utils
from nn.losses import EDLLoss
from nn.model_base import VDANN

# from nn.metrics import CategoricalTruePositives
from pre_process.pre_process import PreProcess
from pre_process.record import Record
from pre_process.utils import preprocess_images, set_seed

# from wandb.keras import WandbCallback
# 環境設定
set_seed(0)
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
EPOCHS = 1
HAS_ATTENTION = True
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
IS_MNIST = True
IS_SIMPLE_ARCH = True
GAMMA = 1
ALPHA = 0
BETA = 0
DROPOUT_RATE = 0.2
BATCH_SIZE = 32
N_CLASS = 10
SBJ_DIM = 68
# KERNEL_SIZE = 512
# KERNEL_SIZE = 256
KERNEL_SIZE = 128
STRIDE = 16
# STRIDE = 16
SAMPLE_SIZE = 10000
LATENT_DIM = 4
NUM_EXAMPLES_TO_GENERATE = 16
INPUT_SHAPE = (28, 28, 1) if IS_MNIST else (64, 30, 1)
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

utils = Utils(
    is_normal=IS_NORMAL,
    is_previous=IS_PREVIOUS,
    data_type=DATA_TYPE,
    fit_pos=FIT_POS,
    stride=STRIDE,
    kernel_size=KERNEL_SIZE,
    model_type="",
    cleansing_type=CLEANSING_TYPE,
)


# データセットの作成
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = preprocess_images(x_train)
x_test = preprocess_images(x_test)
train_size = 60000
batch_size = BATCH_SIZE
test_size = 10000
train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(train_size)
    .batch(batch_size)
)
test_dataset = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .shuffle(test_size)
    .batch(batch_size)
)
data_type = "spectrogram"
shape = (28, 28, 1)
inputs = tf.keras.Input(shape=shape)

epochs = EPOCHS
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = LATENT_DIM
num_examples_to_generate = NUM_EXAMPLES_TO_GENERATE

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim]
)
model = VDANN(
    inputs=inputs,
    gamma=GAMMA,
    latent_dim=latent_dim,
    alpha=ALPHA,
    beta=BETA,
    target_dim=N_CLASS,
    subject_dim=SBJ_DIM,
    has_inception=HAS_INCEPTION,
    has_attention=HAS_ATTENTION,
    is_simple_arch=IS_SIMPLE_ARCH,
    is_mnist=IS_MNIST,
)
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-4),
# )

train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(train_size)
    .batch(batch_size)
)

for example in train_dataset.take(1):
    tmp = example

# Pick a sample of the test set for generating output images
assert batch_size >= num_examples_to_generate
for test_batch_x, test_batch_y in test_dataset.take(1):
    test_sample_x = test_batch_x[0:num_examples_to_generate, :, :, :]
    test_sample_y = test_batch_y[0:num_examples_to_generate]

utils.show_true_image(test_sample_x)

utils.generate_and_save_images(model, 0, test_sample_x)

# model_baes内にGPUの計算中にnumpyに渡すことが出来ないのでmainにlambda式用意
tensor2numpy = lambda x: x.numpy()
vae_opt = tf.keras.optimizers.Adam(1e-4)
tar_opt = tf.keras.optimizers.Adam(1e-4)
for epoch in range(epochs):
    vae_test_loss_metric = tf.keras.metrics.Mean()
    tar_test_loss_metric = tf.keras.metrics.Mean()
    # print("Start of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, data in enumerate(train_dataset):
        train_loss = model.train_step(data, vae_opt, tar_opt)
        # if step % 50 == 0:
        #     print(
        #         f"train loss: (vae, sbj, tar) =  {tuple(map(tensor2numpy, train_loss))}"
        #     )
    # train_metrics.reset_states()

    for step, data in enumerate(test_dataset):
        test_loss = model.test_step(data)
        vae_test_loss_metric(test_loss[0])
        tar_test_loss_metric(test_loss[1])
    vae_elbo = vae_test_loss_metric.result()
    tar_elbo = tar_test_loss_metric.result()
    display.clear_output(wait=False)
    # print(
    #     f"test loss: (vae, sbj, tar) = {tuple(map(tensor2numpy, test_loss))}"
    # )
    print(f"Epoch: {epoch}, VAE ELBO: {vae_elbo}, TARGET LOSS: {tar_elbo}")
    # test_metrics.reset_states()
    utils.generate_and_save_images(model, epoch, test_sample_x)

# gifの作成
utils.create_gif()
# 潜在変数空間を描画
utils.drow_latent_space(model, x_test, y_test)
