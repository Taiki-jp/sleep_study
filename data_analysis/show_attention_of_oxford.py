import os, sys
# import numpy as np
# /nn にいる場合は /data_analysis を追加する必要がある
import tensorflow as tf
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.keras.backend.set_floatx('float32')
import matplotlib.pyplot as plt
from utils import PreProcess
import numpy as np

# TODO : modelDirPath の変更
# modelDirPath = ""
modelFilePath = "sleep_model/20210114-154848"
path = modelDirPath + modelFilePath

model = tf.keras.models.load_model(path)

# 入力と出力を決める
new_input = model.input
# -4(-13, -12, -6, -7) の時は attention を表示する
new_output = model.layers[-6].output

# 上の入出力を持つネットワーク作成
# *** 現在の実装ではエラーが出るため使えない *** <= 修正済み
new_model = tf.keras.Model(new_input, new_output)

new_model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer = tf.keras.optimizers.Adam(),
                  metrics = ["accuracy"])

# 前処理のオブジェクト作成
m_preProcess = PreProcess()

# データ拡張を行うためにジェネレータ作成
trainGenerator, validationGenerator, testGenerator = m_preProcess.make_generator()

# attention 層のニューロン
train_attention = new_model.predict(trainGenerator)
test_attention = new_model.predict(validationGenerator)

# 以下のループ処理で使用するため
bs, H, W, C = train_attention.shape

# =================================================== #
#         ジェネレータを使ったバッチ処理 （ひとまずこれで．本当はラベルを全体の真ん中に持ってきたい）             
# =================================================== #
data, label = next(trainGenerator)
train_attention = new_model.predict(data)
probs = model.predict(data)
probs = tf.math.softmax(probs)
predictedLabels = np.argmax(probs, axis = 1)

for counter, image in enumerate(train_attention):
    # 先にコンフィデンスの処理を行う
    predictedLabel = predictedLabels[counter]
    trueLabel = int(label[counter])
    confidence = probs[counter][trueLabel]
    # グラフ作成
    plt.clf()
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(image, aspect = 'auto')
    ax1.axis('off')
    cbar = fig.colorbar(im)
    ax2 = fig.add_subplot(122)
    ax2.imshow(data[counter], aspect = 'auto')
    ax2.axis('off')
    ax1.set_title(f"conf : {confidence:.1%}, pred label : {predictedLabel}, true label : {trueLabel}", loc = 'left')
    # TODO : path1 の設定
    path1 = ""
    #if not os.path.exists(path1):
    #    os.mkdir(path1)
    #plt.savefig(f'{path1}/{counter}_{file_path_specifier}.png')
    plt.show()

# =================================================== #    
#     様々な画像に対してアテンションを可視化する
# =================================================== #
for i in range(16):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(121)
    # アテンションマスクの可視化
    im = plt.imshow(train_attention[i], aspect = "auto")
    # output をアダマール積後の時はここを使う
    #im = plt.imshow(train_attention[i][:, :, 4:7], aspect = 'auto')
    cbar = fig.colorbar(im)
    ax = fig.add_subplot(122)
    im = plt.imshow(trainGenerator[0][0][i], aspect = 'auto')
    # dbar = fig.colorbar(im)
    plt.title(y_train[i])
    plt.show()

# =================================================== #
# 一枚の画像に対してアテンションのすべてのチャンネルを可視化する
# =================================================== #
for i in range(C):
    id = 2
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(121)
    im = plt.imshow(train_attention[id][:, :, i], aspect = 'auto')
    cbar = fig.colorbar(im)
    ax = fig.add_subplot(122)
    im = plt.imshow(x_train[id], aspect = 'auto')
    # 確率(confidence)を計算
    output = model.predict(tf.expand_dims(x_train[id], 0))
    prob = tf.nn.softmax(output)
    true_label_prob = prob[0, y_train[id]]
    max_label_prob = max(prob[0])
    plt.show()
    print(f"max conf and label : {max_label_prob}, {np.argmax(prob[0])}")
    print(f"true conf and label : {true_label_prob}, {y_train[id]}")

# =================================================== #
# 同じ画像が出力されているかの確認用
# =================================================== #
for id in range(0, 100, 10):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(121)
    im = plt.imshow(train_attention[id][:, :, 0], aspect = 'auto')
    cbar = fig.colorbar(im)
    ax = fig.add_subplot(122)
    im = plt.imshow(x_train[id], aspect = 'auto')
    # 確率(confidence)を計算
    output = model.predict(tf.expand_dims(x_train[id], 0))
    prob = tf.nn.softmax(output)
    true_label_prob = prob[0, y_train[id]]
    max_label_prob = max(prob[0])
    plt.savefig(f"tmp_figure/{id}_second")
    print(f"max conf and label : {max_label_prob}, {np.argmax(prob[0])}")
    print(f"true conf and label : {true_label_prob}, {y_train[id]}")