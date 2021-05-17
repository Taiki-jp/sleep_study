import os
from re import X
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
import random
import numpy as np

fd = FindsDir("sleep")
# input_shape = (batch, vec_dim, channel)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

check1d = True
if check1d:
    model = MyInceptionAndAttentionAnd1dCNN(n_classes=5,
                                            vec_dim=1,
                                            timesteps=512,
                                            batch=32,
                                            findsDirObj=fd)
    input_shape = (2000, 512, 1)
    pse_input = tf.random.normal(input_shape)
    pse_output = [random.randint(0, 5) for _ in range(2000)]
    pse_output = np.array(pse_output)
    print(pse_input.shape)
else:
    model = MyInceptionAndAttention(5, hight=128, width=512, findsDirObj=fd)
    input_shape = (32, 512, 128, 1)
    pse_input = tf.random.normal(input_shape)
    print(pse_input.shape)
# model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# これはうまくいかない
model.fit(x=pse_input, y=pse_output, batch_size=32, epochs=10)

# これはうまくいく
# y_pred = model(pse_input)
# これはうまくいかない
#_ev = model.evaluate(x=pse_input, y=pse_output)

# _pr = model.predict(x=pse_input)

#print(model.summary())