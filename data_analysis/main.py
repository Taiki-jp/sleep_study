# ================================================ #
# *         Import Some Libraries
# ================================================ #

from my_setting import FindsDir, SetsPath
SetsPath().set()
import os, sys
import tensorflow as tf
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.keras.backend.set_floatx('float32')
from utils import PreProcess, Utils
from glob import glob
from pprint import pprint
import numpy as np

# ================================================ #
# *          モデルの読み込みと作成
# ================================================ #

# TODO : modelDirPath の設定
modelDirPath = ""
modelList = glob(modelDirPath+'*')
print("*** this is model list ***")
pprint(modelList)
print("一番新しいモデルが最後に来ていることを確認")
model = tf.keras.models.load_model(modelList[-1])
# 入力と出力を決める
new_input = model.input
new_output = model.get_layer('my_attention2d_4').output
new_model = tf.keras.Model(new_input, new_output)

# ================================================ #
# *              モデルのコンパイル
# ================================================ #

new_model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer = tf.keras.optimizers.Adam(),
                  metrics = ["accuracy"])

# ================================================ #
# *                データの作成
# ================================================ #

SLEEP_STAGE = 5
m_findsDir = FindsDir("sleep")
#inputFileName = input("*** 被験者データを入れてください *** \n")
m_preProcess = PreProcess(project=m_findsDir.returnDirName(), 
                          input_file_name=Utils().name_dict)
(train, test) = m_preProcess.loadData(is_split=True)
(x_train, y_train), (x_test, y_test) = m_preProcess.makeDataSet(train=train, 
                                                                test=test, 
                                                                is_split=True, 
                                                                target_ss=SLEEP_STAGE)
m_preProcess.maxNorm(x_train)
m_preProcess.maxNorm(x_test)
(x_train, y_train) = m_preProcess.catchNone(x_train, y_train)
(x_test, y_test) = m_preProcess.catchNone(x_test, y_test)
y_train = m_preProcess.binClassChanger(y_train, SLEEP_STAGE)  # Counter({2: 346, 1: 2975, 0: 159, 3: 1105, 4: 458})
y_test = m_preProcess.binClassChanger(y_test, SLEEP_STAGE)  # Counter({2: 49, 1: 365, 4: 41, 0: 22, 3: 79})

# nr34:155, nr2: 395, nr1: 37, rem: 165, wake: 41

non_target = list()
target = list()

for num, ss in enumerate(y_train):
    if ss == 0:
        non_target.append(x_train[num])
    elif ss == 1:
        target.append(x_train[num])

non_target = np.array(non_target)
target = np.array(target)

attentionArray = []
confArray = []

convertedArray = [non_target, target]

for num, inputs in enumerate(convertedArray):
    attention = new_model.predict(inputs)
    if num == 0:
        labelNum = 0
    elif num == 1:
        labelNum = 1
    else:
        labelNum = None
    conf = tf.math.softmax(model.predict(inputs))[:, labelNum]
    attentionArray.append(attention)
    confArray.append(conf)

# TODO : pathRoot の設定
pathRoot = ""
savedDirList = ["non_target/", "target/"]
savedDirList = [pathRoot + savedDir for savedDir in savedDirList]

for num, target in enumerate(attentionArray):
    # ANCHOR : なんでかpathがたくさん作られるのでとりあえずコメントアウト
    # m_preProcess.checkPath(savedDirList[num])
    m_preProcess.simpleImage(image_array = target,
                             row_image_array = convertedArray[num],
                             file_path = savedDirList[num],
                             x_label = "time[s]",
                             y_label = "frequency[Hz]",
                             title_array = confArray[num])

