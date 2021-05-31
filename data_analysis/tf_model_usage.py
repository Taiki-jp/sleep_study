import tensorflow as tf
import matplotlib.pyplot as plt
import os, sys, glob
from pprint import pprint
import numpy as np
import seaborn as sns
import scipy.stats
import pandas as pd

# 移動
if os.getcwd() != os.path.join(os.environ["USERPROFILE"], "/git/sleepstudy/wandb"):
    os.chdir(os.environ["USERPROFILE"] + "/git/sleepstudy/wandb")

# 移動確認
print(os.getcwd())

# 読み込むファイル名のリスト作成
modelList = glob.glob("run-20201216*")

# 保存用のリスト作成
freqMeanList = np.array([])
freqAbsMeanList = np.array([])
freqVarList = np.array([])

# 各被験者データに対して読み込む
for model in modelList:
    filePath = os.path.join(model, "files")
    try:
        model = tf.keras.models.load_model(filePath+"/model.h5")
    except:
        continue
    freq = model.weights[0].numpy()
    row, col = freq.shape
    if row != 512:
        continue
    else:
        freqMean = freq.mean(axis = 1)
        freqAbs = np.abs(freq)
        freqVar = freq.std(axis = 1)
        freqAbsMean = freqAbs.mean(axis = 1)
        freqMeanList = np.append(freqMeanList, freqMean)
        freqAbsMeanList = np.append(freqAbsMeanList, freqAbsMean)
        freqVarList = np.append(freqVarList, freqVar)

# np.append で 1 次元になっているので 2 次元に変換
tmp = freqVarList.reshape(512, -1)
# 正規化
tmp = scipy.stats.zscore(tmp, axis = 1)
# 全ての被験者に対して平均を取る
# tmp = np.mean(tmp, axis = 1)
# データフレームに入れる
x = np.linspace(0, 8, 512)
intX = [int(i) for i in x]
df = pd.DataFrame(tmp,
                  index = intX)

# heatmap 作成
sns.heatmap(df,
            # xticklabels = 100,
            yticklabels = 64)

# TODO : filePath を d:の下に持っていく
filePath = ""
plt.savefig(filePath)


plt.clf()
# np.append で 1 次元になっているので 2 次元に変換
tmp = freqMeanList.reshape(512, -1)
# 正規化
tmp = scipy.stats.zscore(tmp, axis = 1)
# 全ての被験者に対して平均を取る
# tmp = np.mean(tmp, axis = 1)
# データフレームに入れる
x = np.linspace(0, 8, 512)
intX = [int(i) for i in x]
df = pd.DataFrame(tmp,
                  index = intX)

# heatmap 作成
sns.heatmap(df,
            # xticklabels = 100,
            yticklabels = 64)

# TODO : filePath の設定
filePath = ""
plt.savefig(filePath)


plt.clf()
# np.append で 1 次元になっているので 2 次元に変換
tmp = freqAbsMeanList.reshape(512, -1)
# 正規化
tmp = scipy.stats.zscore(tmp, axis = 1)
# 全ての被験者に対して平均を取る
# tmp = np.mean(tmp, axis = 1)
# データフレームに入れる
x = np.linspace(0, 8, 512)
intX = [int(i) for i in x]
df = pd.DataFrame(tmp,
                  index = intX)

# heatmap 作成
sns.heatmap(df,
            # xticklabels = 100,
            yticklabels = 64)

# TODO : filePath の設定
filePath = ""
plt.savefig(filePath)