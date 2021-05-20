import tensorflow as tf
import matplotlib.pyplot as plt
import os, sys, glob
from pprint import pprint
import numpy as np
import seaborn as sns
import scipy.stats
import pandas as pd
sys.path.append("c:/users/takadamalab/git/sleep_study/nn")
sys.path.append("c:/users/takadamalab/git/sleep_study/pre_process")
from losses import EDLLoss
from utils import FindsDir, PreProcess, Utils
from load_sleep_data import LoadSleepData
from my_model import MyInceptionAndAttention

# float32が推奨されているみたい
tf.keras.backend.set_floatx('float32')
# tf.functionのせいでデバッグがしずらい問題を解決してくれる（これを使わないことでエラーが起こらなかったりする）
tf.config.run_functions_eagerly(True)

# murakamiテストのときのENNのパス
path = os.path.join(os.environ["sleep"], "models", "20210520-194451")

# モデル読み込み
model = tf.keras.models.load_model(path)
                                   #custom_objects={"EDLLoss":EDLLoss(K=5)},
                                   #compile=False)

# うまく読み込めないときに使える（keras.Model型ではないのでfitとかはできない）

# コンパイル
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=EDLLoss(K=5, batch_size=32),
              metrics=["accuracy"])

# 評価（訓練データ）
fd = FindsDir("sleep")
m_preProcess = PreProcess(input_file_name=Utils().name_dict)
m_loadSleepData = LoadSleepData(input_file_name="H_Li")  # TODO : input_file_nameで指定したファイル名はload_data_allを使う際はいらない
MUL_NUM = 1
is_attention = True
attention_tag = "attention" if is_attention else "no-attention"
datasets = m_loadSleepData.load_data_all()
# TODO : テストの人に応じてここを変えて
test_id = 1
(train, test) = m_preProcess.split_train_test_from_records(datasets, test_id=test_id)
(x_train, y_train), (x_test, y_test) = m_preProcess.makeDataSet(train=train, 
                                                                test=test, 
                                                                is_set_data_size=True,
                                                                mul_num=1,
                                                                is_storchastic=False) 
y_train = tf.one_hot(y_train, depth=5)
print(x_train[0].shape)
pse_data = tf.expand_dims(x_train[0], axis=2)
pse_data.shape

model.predict(x_train)