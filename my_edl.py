import os
from tensorflow.keras import metrics
from nn.my_setting import SetsPath, FindsDir
SetsPath().set()
import datetime, wandb
from pre_process.wandb_classification_callback import WandbClassificationCallback
import tensorflow as tf
from nn.my_model import MyInceptionAndAttention, MyInceptionAndAttentionAnd1dCNN
from losses import EDLLoss
from pre_process.load_sleep_data import LoadSleepData
from pre_process.utils import PreProcess, Utils
from collections import Counter

fd = FindsDir("sleep")
# input_shape = (batch, vec_dim, channel)

check1d = False
if check1d:
    model = MyInceptionAndAttentionAnd1dCNN(n_classes=5,
                                            vec_dim=1,
                                            timesteps=512,
                                            batch=32,
                                            findsDirObj=fd)
    input_shape = (2000, 512, 1)
    pse_input = tf.random.normal(input_shape)
    print(pse_input.shape)
else:
    model = MyInceptionAndAttention(5, hight=128, width=512, findsDirObj=fd)
    input_shape = (32, 512, 128, 1)
    pse_input = tf.random.normal(input_shape)
    print(pse_input.shape)
# model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy', 'mse'])

model(pse_input)

#print(model.summary())