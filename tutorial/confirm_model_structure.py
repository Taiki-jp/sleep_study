# モデルの学習構造がしっかりできていることを確認する
import os
from tensorflow.python.keras.backend import shape
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from nn.model_base import EDLModelBase
from nn.my_setting import SetsPath, FindsDir
SetsPath().set()
import tensorflow as tf
from nn.my_model import MyInceptionAndAttention
from losses import EDLLoss
import numpy as np
import random

# NOTE : gpuを設定していない環境のためにエラーハンドル
try:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("GPUがサポートされていません")

# float32が推奨されているみたい
tf.keras.backend.set_floatx('float32')
# tf.functionのせいでデバッグがしずらい問題を解決してくれる（これを使わないことでエラーが起こらなかったりする）
#tf.config.run_functions_eagerly(True)

# (2000, 512, 128, 1)のサイズの仮入力データの作成
def make_pse_data(type="sleep_data_2d"):
    input_shape = (2000, 512, 128, 1)
    x = tf.random.normal(input_shape)
    y = [random.randint(0, 5) for _ in range(2000)]
    y = np.array(y)
    y = tf.one_hot(y, depth=5)
    return x, y

# function APIによるモデル構築
def edl_classifier_2d(x, n_class):
    # convolution AND batch normalization
    def _conv2d_bn(x, filters, num_row, num_col,
                   padding='same', strides=(1,1),name=None):
        if name is not None:
            bn_name = name+'_bn'
            conv_name = name+'_conv'
        else:
            bn_name = None
            conv_name = None
        x = tf.keras.layers.Conv2D(filters, (num_row, num_col),
                                   strides=strides,
                                   padding=padding,
                                   use_bias=False,
                                   name=conv_name)(x)
        x = tf.keras.layers.BatchNormalization(scale=False, name=bn_name)(x)
        x = tf.keras.layers.Activation('relu', name=name)(x)
        return x

    # shapeが合うように調整
    x = tf.keras.layers.Conv2D(3, (1, 4), (1, 4))(x)
    x = tf.keras.layers.Activation('relu')(x)
    # 畳み込み開始01
    x = _conv2d_bn(x, 32, 3, 3, strides=(2,2), padding='valid')
    x = _conv2d_bn(x, 32, 3, 3, padding='valid')
    x = _conv2d_bn(x, 64, 3, 3)
    x = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2))(x)
    # 畳み込み開始02
    x = _conv2d_bn(x, 80, 1, 1, padding='valid')
    x = _conv2d_bn(x, 192, 3, 3, padding='valid')
    x = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2))(x)
    
    # mixed 1: 35 x 35 x 288
    branch1x1 = _conv2d_bn(x, 64, 1, 1)  
    branch5x5 = _conv2d_bn(x, 48, 1, 1)
    branch5x5 = _conv2d_bn(branch5x5, 64, 5, 5)  
    branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)    
    branch_pool = tf.keras.layers.AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = _conv2d_bn(branch_pool, 64, 1, 1)
    x = tf.keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                                    axis=-1, name='mixed1')
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(n_class**2)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(n_class)(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

# フォルダ見つける
fd = FindsDir("sleep")

inputs = tf.keras.Input(shape=(512, 128, 1))
outputs = edl_classifier_2d(x=inputs, n_class=5)

model = EDLModelBase(inputs=inputs, outputs=outputs)

# コンパイル
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=EDLLoss(K=5),
              metrics=['accuracy'])

# データ取得
x, y = make_pse_data()

# モデルの学習
model.fit(x, y, epochs=50)

# モデルの保存
model.save("model_saved_by_save_method")

# 保存したモデルの読み込み
loaded_model = tf.keras.models.load_model("model_saved_by_save_method/",
                                          custom_objects={"EDLLoss":EDLLoss(K=5,annealing=0.1)})

# モデルの評価
loaded_model.evaluate(x, y)