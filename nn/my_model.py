# ================================================ #
# *         Import Some Libraries
# ================================================ #

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras import backend

import nn.layer_base as MyLayer
from nn.model_base import CreateModelBase, EDLModelBase, ModelBase


def Int2IntWithSequentialModel(hidded1_dim, hidden2_dim):
    """スカラからスカラを予測する全結合モデル

    Args:
        hidded1_dim ([int]): [一層目のユニット数]
        hidden2_dim ([int]): [二層目のユニット数]

    Returns:
        [model]: [モデルを返す]
    """
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(hidded1_dim, activation="relu", input_shape=(1,))
    )
    model.add(tf.keras.layers.Dense(hidden2_dim, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    print(model.summary)
    return model


# 複数出力を持つ関数の書き方
def build_model():
    inputs = tf.keras.Input(shape=(10,))
    first = tf.keras.layers.Dense(units=10, activation="relu")(inputs)
    second = tf.keras.layers.Dense(units=10, activation="relu")(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=[first, second])
    return model


# ================================================ #
# *           FNN を使ったモデル
# TODO : サブクラスの実装はどうしよう
# ================================================ #


class MyFnnModel(CreateModelBase):
    def __init__(
        self,
        input_dim,
        hidden_dim1,
        hidden_dim2,
        n_classes,
        load_file_path=None,
    ):
        super().__init__(load_file_path=load_file_path)
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.n_classes = n_classes
        self.inputs = tf.keras.Input(self.input_dim)
        self.dense1 = tf.keras.layers.Dense(self.hidden_dim1)
        self.dense2 = tf.keras.layers.Dense(self.hidden_dim2)
        self.dense3 = tf.keras.layers.Dense(self.n_classes)
        self.model = self.createModel()

    def createModel(self, by_functional=False):
        if by_functional:
            pass
        else:
            model = tf.keras.Sequential()
            model.add(self.inputs)
            model.add(self.dense1)
            model.add(self.dense2)
            model.add(self.dense3)
            return model


# ================================================ #
# *           Conv1D を使ったモデル
# TODO : サブクラスの実装はどうしよう
# ================================================ #


class MyConv1DModel(CreateModelBase):
    def __init__(
        self,
        n_classes,
        load_file_path=None,
        findsDirObj=None,
        with_attention=False,
    ):
        super().__init__(
            load_file_path=load_file_path, findsDirObj=findsDirObj
        )
        self.n_classes = n_classes
        # ! NOTE : input_shape = (batch, timesteps, channel)
        self.input_shape = (32, 512, 1)
        # ! NOTE : self.input_shape[1:] は (timesteps, channel) の２次元を返す(ここでは(512, 1))
        self.inputs = tf.keras.Input(shape=self.input_shape[1:])
        self.conv1 = tf.keras.layers.Conv1D(
            filters=64, kernel_size=3, strides=3, activation="relu"
        )
        self.maxpool1 = tf.keras.layers.MaxPool1D()
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.conv2 = tf.keras.layers.Conv1D(
            filters=128, kernel_size=3, strides=3, activation="relu"
        )
        self.maxpool2 = tf.keras.layers.MaxPool1D()
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.GAP = tf.keras.layers.GlobalAveragePooling1D()
        self.dense1 = tf.keras.layers.Dense(128)
        self.dense2 = tf.keras.layers.Dense(self.n_classes)
        if with_attention:
            self.with_attention = with_attention
            self.attention = MyLayer.MyAttention1D()
        self.model = self.createModel()
        pass

    def createModel(self, by_function=True):
        if by_function:
            x = self.conv1(self.inputs)
            x = self.maxpool1(x)
            x = self.dropout1(x)
            x = self.conv2(x)
            x = self.maxpool2(x)
            x = self.dropout2(x)
            if self.with_attention:
                attention = self.attention(x)
                x *= attention
            x = self.GAP(x)
            x = self.dense1(x)
            x = self.dense2(x)
            return tf.keras.Model(self.inputs, x)
        else:
            if self.with_attention:
                print(
                    'Now you are building with sequential with attention. \
                      You should change "by_function = True" to avoid sequential model'
                )
            model = tf.keras.Sequential()
            model.add(self.inputs)
            model.add(self.conv1)
            model.add(self.maxpool1)
            model.add(self.dropout1)
            model.add(self.conv2)
            model.add(self.maxpool2)
            model.add(self.dropout2)
            model.add(self.GAP)
            model.add(self.dense1)
            model.add(self.dense2)
            return model


# ================================================ #
# *     Functional x ２次元畳み込みモデル
# ================================================ #


class MyConv2DModelUnused(ModelBase):
    def __init__(
        self,
        epochs,
        autoCompile=True,
        row_size=32,
        column_size=512,
        channel_origin=1,
        filter_01=4,
        filter_02=4,
        filter_03=4,
    ):
        """2次元畳み込み NN の初期化メソッド

        Args:
            autoCompile (bool, optional): [コンパイル方法を指定したいときは False にしてこのクラス内のコンパイルメソッドに変更を加える]. Defaults to True.
        """
        super().__init__()
        self.columnSize = column_size
        self.rowSize = row_size
        self.channel_origin = channel_origin
        self.channel_01 = filter_01
        self.channel_02 = filter_02
        self.channel_03 = filter_03
        self.kernelSize_01 = channel_origin
        self.batchSize = 16
        self.epochs = epochs
        self.verbose = 2
        self.autoCompile = autoCompile

    def call(self, x):
        # input の shape は (row size, column size, channel size) にする？
        x = tf.keras.layers.Conv2D(16, kernel_size=(1, 2), strides=(1, 2))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(x)
        x = tf.keras.layers.Conv2D(32, kernel_size=(1, 2), strides=(1, 2))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(x)
        x = tf.keras.layers.Conv2D(128, kernel_size=(2, 2), strides=(2, 2))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        outputs = tf.keras.layers.Dense(5)(x)
        return outputs


# ================================================ #
# *     Inception を使った簡単なモデル
# *                Oxford-ped
# (num_classes, hight, width, channel) = (37, 224, 224, 3)
# *                  MNIST
# (num_classes, hight, width, channel) = (10, 28, 28, 1)
# ================================================ #


class MyInception(ModelBase):
    def __init__(self, num_classes, hight, width, channel):
        super().__init__()
        self.baseModel = tf.keras.applications.InceptionV3(
            include_top=False, input_shape=(hight, width, channel)
        )
        self.GAP = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        baseInputs = self.baseModel.layers[0].input
        outputs = self.baseModel.layers[-1].output
        outputs = self.GAP(outputs)
        outpus = self.dense1(outputs)
        return outputs


# ================================================ #
# *         SubClass x Inception モデル
# ================================================ #


class MyInceptionAndAttention(tf.keras.Model):
    def __init__(
        self,
        n_classes,
        hight,
        width,
        findsDirObj,
        channel=1,
        is_attention=True,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.hight = hight
        self.width = width
        self.channel = channel
        self.conv = tf.keras.layers.Conv2D(
            filters=3, kernel_size=(1, 4), strides=(1, 4)
        )
        self.baseModel = tf.keras.applications.InceptionV3(include_top=False)
        self.baseInputs = self.baseModel.layers[0].input
        # FIXME : gradientが存在しない層が複数個存在する（なぜ？）
        self.baseOutputs = self.baseModel.get_layer("mixed0").output
        self.feature = tf.keras.Model(self.baseInputs, self.baseOutputs)
        self.attention = MyLayer.MyAttention2D(filters=1, kernel_size=1)
        self.GAP = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(n_classes ** 2)
        self.dense2 = tf.keras.layers.Dense(n_classes)
        self.is_attention = is_attention

    def call(self, x):
        # xが3次元データのときは以下の処理を入れる
        x = self.conv(x)
        x = self.feature(x)
        if self.is_attention:
            attention = self.attention(x)
            x *= attention
        x = self.GAP(x)
        x = self.dense1(x)
        x = self.dense2(x)
        evidence = tf.nn.relu(x)
        return evidence


# ================================================ #
# *         Functional x Inception モデル
# !           NOTE : 今一番いいやつ
# ================================================ #


class MyInceptionAndAttentionWithoutSubClassing(CreateModelBase):
    def __init__(
        self,
        n_classes,
        hight,
        width,
        channel,
        file_path=None,
        weights="imagenet",
    ):
        super().__init__(load_file_path=file_path)
        self.baseModel = tf.keras.applications.InceptionV3(
            include_top=False,
            input_shape=(hight, width, channel),
            weights=weights,
        )

        self.GAP = tf.keras.layers.GlobalAveragePooling2D()
        self.attention1 = MyLayer.MyAttention2D(filters=1, kernel_size=1)
        self.attention2 = MyLayer.MyAttention2D(filters=1, kernel_size=1)
        self.attention3 = MyLayer.MyAttention2D(filters=1, kernel_size=1)
        self.attention4 = MyLayer.MyAttention2D(filters=1, kernel_size=1)
        self.dense1 = tf.keras.layers.Dense(n_classes)
        self.concat = tf.concat
        self.file_path = file_path
        pass

    def createModel(self, output_layer):
        # 入力と特徴量抽出
        baseInputs = self.baseModel.layers[0].input
        baseOutputs = self.baseModel.layers[output_layer].output
        # アテンション
        attention1 = self.attention1(baseOutputs)
        attention2 = self.attention2(baseOutputs)
        # attention3 = self.attention3(baseOutputs)
        # attention4 = self.attention4(baseOutputs)
        # アダマール積
        baseOutputs1 = baseOutputs * attention1
        baseOutputs2 = baseOutputs * attention2
        # baseOutputs3 = baseOutputs * attention3
        # baseOutputs4 = baseOutputs * attention4
        # baseOutputs = baseOutputs1 # * baseOutputs2 #  * baseOutputs3 * baseOutputs4)
        # GAP で1次元に戻す
        baseOutputs1 = self.GAP(baseOutputs1)
        baseOutputs2 = self.GAP(baseOutputs2)
        # 結合
        # NOTE :
        baseOutputs = self.concat([baseOutputs1, baseOutputs2], axis=1)
        # 全結合でＮクラス分類（ソフトマックスは挟まない）
        baseOutputs = self.dense1(baseOutputs)
        # モデルを作成して返す
        return tf.keras.Model(baseInputs, baseOutputs)


# ================================================ #
# *         Conv2D x Attention モデル
# ================================================ #


class MyConv2DModel(CreateModelBase):
    def __init__(
        self,
        n_classes,
        hight,
        width,
        channel,
        file_path=None,
        with_attention=False,
    ):
        super().__init__(load_file_path=file_path)
        self.n_classes = n_classes
        # ! NOTE : input_shape = (batch, timesteps, channel)
        self.input_shape = (hight, width, channel)
        # ! NOTE : self.input_shape[1:] は (timesteps, channel) の２次元を返す(ここでは(512, 1))
        self.inputs = tf.keras.Input(shape=self.input_shape)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(1, 2), activation="relu"
        )
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(1, 2))
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=3, strides=(1, 2), activation="relu"
        )
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(1, 2))
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.GAP = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(128)
        self.dense2 = tf.keras.layers.Dense(self.n_classes)
        if with_attention:
            self.with_attention = with_attention
            self.attention = MyLayer.MyAttention2D(filters=1, kernel_size=1)
        pass

    def createModel(self, by_function=True):
        if by_function:
            x = self.conv1(self.inputs)
            x = self.maxpool1(x)
            x = self.dropout1(x)
            x = self.conv2(x)
            x = self.maxpool2(x)
            x = self.dropout2(x)
            if self.with_attention:
                attention = self.attention(x)
                x *= attention
            x = self.GAP(x)
            x = self.dense1(x)
            x = self.dense2(x)
            return tf.keras.Model(self.inputs, x)
        else:
            if self.with_attention:
                print(
                    'Now you are building with sequential with attention. \
                      You should change "by_function = True" to avoid sequential model'
                )
            model = tf.keras.Sequential()
            model.add(self.inputs)
            model.add(self.conv1)
            model.add(self.maxpool1)
            model.add(self.dropout1)
            model.add(self.conv2)
            model.add(self.maxpool2)
            model.add(self.dropout2)
            model.add(self.GAP)
            model.add(self.dense1)
            model.add(self.dense2)
            return model


# ================================================ #
# *         resnet50 x Attention モデル
# ================================================ #


class MyResNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.base_model = ResNet50(input_shape=(128, 512, 1))
        inputs = self.base_model.input
        outputs = self.base_model.layers[-3].output
        self.features = tf.keras.Model(inputs, outputs)
        # ↓ (none, 7, 7, 2048)
        _, H, W, C = outputs.shape
        self.attn_conv = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(H, W, C)),
                tf.keras.layers.Conv2D(
                    filters=4, kernel_size=5, padding="same", activation="relu"
                ),
            ]
        )
        # ↓ (none, 7, 7, 4)
        self.fc = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(7, 7, 4)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(37),
            ]
        )

    def call(self, x):
        x = self.features(x)
        attn = self.attn_conv(x)
        x *= attn
        x = self.fc(x)
        return x


# ================================================ #
# *        データの成型だけを行うモデル
# ================================================ #


class MyShapingModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            3, (1, 4), (1, 4), activation="relu"
        )

    def call(self, x):
        x = self.conv(x)
        return x

    def create(self):
        inputs = tf.keras.Input(shape=(128, 512, 1))
        return tf.keras.Model([inputs], self.call(inputs))


# ================================================ #
# *      Attention x 1dCNN x Inception Module
# ================================================ #


class MyInceptionAndAttentionAnd1dCNN(tf.keras.Model):
    def __init__(
        self,
        n_classes,
        vec_dim,
        timesteps,
        findsDirObj,
        batch=None,
        is_attention=True,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.vec_dim = vec_dim
        self.timesteps = timesteps
        self.batch = batch
        self.findsDirObj = findsDirObj
        self.is_attentino = is_attention
        if backend.image_data_format() == "channel_first":
            self.channel_axis = 1
        else:
            self.channel_axis = 2
        # input_shape = (batch, timesteps, vector's dim)
        self.my_input_shape = (self.batch, self.timesteps, self.vec_dim)
        self.gap = tf.keras.layers.GlobalAveragePooling1D()
        self.dense1 = tf.keras.layers.Dense(
            self.n_classes ** 2, activation="relu"
        )
        self.dense2 = tf.keras.layers.Dense(self.n_classes)

    def call(self, x):
        x = self.conv1d_bn(
            x=x, filters=32, kernelsize=3, strides=2, padding="valid"
        )
        x = self.conv1d_bn(x=x, filters=32, kernelsize=3, padding="valid")
        x = self.conv1d_bn(x=x, filters=64, kernelsize=3)
        x = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2)(x)

        x = self.conv1d_bn(x=x, filters=80, kernelsize=1, padding="valid")
        x = self.conv1d_bn(x=x, filters=192, kernelsize=3, padding="valid")
        x = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2)(x)
        # mixed 0: 35 x 256
        branch1x1 = self.conv1d_bn(x=x, filters=64, kernelsize=1)

        branch5x5 = self.conv1d_bn(x=x, filters=48, kernelsize=1)
        branch5x5 = self.conv1d_bn(x=x, filters=64, kernelsize=5)

        branch3x3dbl = self.conv1d_bn(x=x, filters=64, kernelsize=1)
        branch3x3dbl = self.conv1d_bn(x=branch3x3dbl, filters=96, kernelsize=3)
        branch3x3dbl = self.conv1d_bn(x=branch3x3dbl, filters=96, kernelsize=3)

        branch_pool = tf.keras.layers.AveragePooling1D(
            pool_size=3, strides=1, padding="same"
        )(x)
        branch_pool = self.conv1d_bn(x=branch_pool, filters=32, kernelsize=1)

        x = tf.keras.layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=self.channel_axis,
            name="mixed0",
        )
        # gather along with channel axis
        x = self.gap(x)
        x = self.dense1(x)
        x = self.dense2(x)
        # evidence : 活性化関数(普通の関数の出力のイメージ)．論文内のe_kの事
        evidence = tf.nn.relu(x)
        # alpha : ディリクレ分布のハイパーパラメータ
        alpha = evidence + 1
        # uncertainty : クラス分類数/Σ alpha
        u = self.n_classes / tf.reduce_sum(alpha, axis=1, keepdims=True)
        # probabitity : クラス予測
        prob = alpha / tf.reduce_sum(alpha, axis=1, keepdims=True)
        return prob

    def createModel(self):
        inputs = tf.keras.Input(shape=self.my_input_shape[1:])
        return tf.keras.Model(inputs, self.call(inputs))

    # TODO : この関数が原因っぽいので参照元と比較したり、関数だけ取り出して学習できるか確かめる
    def conv1d_bn(
        self, x, filters, kernelsize, padding="same", strides=1, name=None
    ):
        if name is not None:
            bn_name = name + "_bn"
            conv_name = name + "_conv"
        else:
            bn_name = None
            conv_name = None
        if backend.image_data_format() == "channels_first":
            bn_axis = 1
        else:
            bn_axis = 1
        x = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernelsize,
            strides=strides,
            padding=padding,
            use_bias=False,
            name=conv_name,
        )(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, scale=False, name=bn_name
        )(x)
        x = tf.keras.layers.Activation("relu", name=name)(x)
        return x


# ================================================ #
# *                モデルのビルト
# ================================================ #


def build_my_model(input_shape: tuple, model: tf.keras.Model):
    inputs = tf.keras.Input(shape=input_shape)
    outputs = model(inputs)
    model = tf.keras.Model(inputs, outputs)
    return model


# ================================================ #
# *         テスト用メイン部分
# ================================================ #

if __name__ == "__main__":
    # model = MyInceptionAndAttention(5, 224, 224, 1).createModel()
    # print(model.summary())
    from data_analysis.utils import FindsDir

    fd = FindsDir("sleep")
    model = build_my_model(
        input_shape=(224, 224, 1),
        model=MyInceptionAndAttention(5, 512, 128, fd),
    )
    print(model.summary())
