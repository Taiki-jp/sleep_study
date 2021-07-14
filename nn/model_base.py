import os, datetime
import numpy as np
import tensorflow as tf

# ================================================ #
#           function APIによるモデル構築(1d-conv)
# ================================================ #


def edl_classifier4psedo_data(x, use_bias, hidden_dim):
    x = tf.keras.layers.Dense(
        hidden_dim, activation="relu", use_bias=use_bias
    )(x)
    x = tf.keras.layers.Dense(
        hidden_dim, activation="relu", use_bias=use_bias
    )(x)
    x = tf.keras.layers.Dense(2, activation="relu", use_bias=use_bias)(x)
    return x


# ================================================ #
#           function APIによるモデル構築(1d-conv)
# ================================================ #
def edl_classifier_1d(x, n_class, has_attention=True, has_inception=True):
    # convolution AND batch normalization
    def _conv1d_bn(x, filters, num_col, padding="same", strides=1, name=None):
        if name is not None:
            bn_name = name + "_bn"
            conv_name = name + "_conv"
        else:
            bn_name = None
            conv_name = None
        x = tf.keras.layers.Conv1D(
            filters,
            num_col,
            strides=strides,
            padding=padding,
            use_bias=False,
            name=conv_name,
        )(x)
        x = tf.keras.layers.BatchNormalization(scale=False, name=bn_name)(x)
        x = tf.keras.layers.Activation("relu", name=name)(x)
        return x

    # start convolution from 512/4 -> 128
    x = tf.keras.layers.Conv1D(3, 4, strides=4, name="shrink_tensor_layer")(x)
    x = tf.keras.layers.Activation("relu")(x)
    # 畳み込み開始01
    x = _conv1d_bn(x, 32, 3, strides=2, padding="valid", name="first_layer")
    x = _conv1d_bn(x, 32, 3, padding="valid", name="second_layer")
    x = _conv1d_bn(x, 64, 3, name="third_layer")
    x = tf.keras.layers.MaxPooling1D(3, strides=2, name="first_max_pool")(x)
    # 畳み込み開始02
    x = _conv1d_bn(x, 80, 1, padding="valid", name="forth_layer")
    x = _conv1d_bn(x, 192, 3, padding="valid", name="fifth_layer")
    x = tf.keras.layers.MaxPooling1D(3, strides=2, name="second_max_pool")(x)

    if has_inception:
        # mixed 1: 35 x 35 x 288
        branch1x1 = _conv1d_bn(x, 64, 1)
        branch5x5 = _conv1d_bn(x, 48, 1)
        branch5x5 = _conv1d_bn(branch5x5, 64, 5)
        branch3x3dbl = _conv1d_bn(x, 64, 1)
        branch3x3dbl = _conv1d_bn(branch3x3dbl, 96, 3)
        branch3x3dbl = _conv1d_bn(branch3x3dbl, 96, 3)
        branch_pool = tf.keras.layers.AveragePooling1D(
            3, strides=1, padding="same"
        )(x)
        branch_pool = _conv1d_bn(branch_pool, 64, 1)
        x = tf.keras.layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=-1,
            name="mixed1",
        )  # (13, 13, 288)
        if has_attention:
            attention = tf.keras.layers.Conv1D(
                1, kernel_size=3, padding="same"
            )(
                x
            )  # (13, 13, 1)
            attention = tf.keras.layers.Activation("sigmoid")(attention)
            x = tf.multiply(x, attention)
            # x *= attention

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(n_class ** 2)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(n_class)(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


# ================================================ #
#           function APIによるモデル構築(2d-conv)
# ================================================ #


def edl_classifier_2d(x, n_class, has_attention=True, has_inception=True):
    # convolution AND batch normalization
    def _conv2d_bn(
        x, filters, num_row, num_col, padding="same", strides=(1, 1), name=None
    ):
        if name is not None:
            bn_name = name + "_bn"
            conv_name = name + "_conv"
        else:
            bn_name = None
            conv_name = None
        x = tf.keras.layers.Conv2D(
            filters,
            (num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=False,
            name=conv_name,
        )(x)
        x = tf.keras.layers.BatchNormalization(scale=False, name=bn_name)(x)
        x = tf.keras.layers.Activation("relu", name=name)(x)
        return x

    # shapeが合うように調整
    x = tf.keras.layers.Conv2D(3, (1, 4), (1, 4))(x)
    x = tf.keras.layers.Activation("relu")(x)
    # 畳み込み開始01
    x = _conv2d_bn(x, 32, 3, 3, strides=(2, 2), padding="valid")
    x = _conv2d_bn(x, 32, 3, 3, padding="valid")
    x = _conv2d_bn(x, 64, 3, 3)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    # 畳み込み開始02
    x = _conv2d_bn(x, 80, 1, 1, padding="valid")
    x = _conv2d_bn(x, 192, 3, 3, padding="valid")
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    if has_inception:
        # mixed 1: 35 x 35 x 288
        branch1x1 = _conv2d_bn(x, 64, 1, 1)
        branch5x5 = _conv2d_bn(x, 48, 1, 1)
        branch5x5 = _conv2d_bn(branch5x5, 64, 5, 5)
        branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch_pool = tf.keras.layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding="same"
        )(x)
        branch_pool = _conv2d_bn(branch_pool, 64, 1, 1)
        x = tf.keras.layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=-1,
            name="mixed1",
        )  # (13, 13, 288)
        if has_attention:
            attention = tf.keras.layers.Conv2D(
                1, kernel_size=3, padding="same"
            )(
                x
            )  # (13, 13, 1)
            attention = tf.keras.layers.Activation("sigmoid")(attention)

            x *= attention

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(n_class ** 2)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(n_class)(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


# ================================================ #
# *     Model を継承した自作のモデルクラス
# ================================================ #


class ModelBase(tf.keras.Model):
    def __init__(self, fd=None):
        """[初期化メソッド]

        Args:
            fd ([type], optional): [モデル保存のパスを見つけるために必要]. Defaults to None.
        """
        super().__init__()
        self.fd = fd
        self.model = None

    # TODO : 2snake_case
    def save_model(self, id):
        """パスを指定しなくていい分便利

        Args:
            id ([string]): [名前が被らないように日付を渡す]
        """
        path = os.path.join(self.fd.returnFilePath(), "models", id)
        self.model.save(path)

    # TODO : 2snake_case
    def autoCompile(self):
        return self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=["accuracy"],
        )


# ================================================ #
# *　Model に対して様々な自動操作を行うクラスをそろえた関数
# ================================================ #


class CreateModelBase(object):
    def __init__(
        self,
        load_file_path,
        fd=None,
        base_model=None,
        exploit_input_layer=0,
        exploit_output_layer=None,
    ):
        """モデル作成のベースクラス

        Args:
            load_file_path ([bool]): [継承先でここにパスを入れると指定したファイルパスのモデルを読み込む]
            fd([object]): [保存パスを見つけるためのオブジェクト] Defaults to None
            base_model ([model], optional): [ベース構造（特徴抽出）に用いたいモデルを入れる]. Defaults to None.
            exploit_input_layer (int, optional): [ベース構造が指定されていないときは0]. Defaults to 0.
            exploit_output_layer ([int], optional): [ベース構造が指定されていないときはNone]. Defaults to None.
        """
        self.load_file_path = load_file_path
        self.fd = fd
        self.time_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.base_model = base_model
        self.exploit_input_layer = exploit_input_layer
        self.exploit_output_layer = exploit_output_layer
        self.model = None
        pass

    def save_model(self, id):
        """パスを指定しなくていい分便利

        Args:
            id ([string]): [名前が被らないように日付を渡す]
        """
        path = os.path.join(self.fd.returnFilePath(), "models", id)
        self.model.save(path)

    def autoCompile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=["accuracy"],
        )

    def utilizeLayer(self, input_layer=None, output_layer=None):
        #! input_layer はデフォルトで０なのでその時は引数を指定して
        #! if 文が無視されるけど構わない
        """TODO : 細かくどうなっているのかは実験を通して確認

        Args:
            input_layer ([type], optional): [description]. Defaults to None.
            output_layer ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if input_layer:
            self.exploit_input_layer = input_layer
        if output_layer:
            self.exploit_output_layer = output_layer
        base_inputs = self.base_model.layers[self.exploit_input_layer].input
        base_outputs = self.base_model.layers[self.exploit_output_layer].output
        return base_inputs, base_outputs


# ================================================ #
# *　Model に対して様々な自動操作を行うクラスをそろえた関数
# ================================================ #


class EDLModelBase(tf.keras.Model):

    # NOTE : model.fitを呼ぶと，train_stepが呼ばれる
    # その結果中身のself(x, training)によってmodel.callが呼ばれる
    # そのためcallメソッド内にtrainingなどの引数を受け取れるように設定しておく必要がある
    def __init__(self, n_class=5, **kwargs):
        super().__init__(**kwargs)
        self.time_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.n_class = n_class

    @tf.function
    def train_step(self, data):
        # x.shape : (32, 128, 512, 1)
        # y.shape : (32,)
        x, y = data

        with tf.GradientTape() as tape:
            # Caclulate predictions
            evidence = self(x, training=True)  # (32, 5)
            alpha = evidence + 1  # (32, 5)
            # uncertainty = self.n_class/tf.reduce_sum(alpha, axis=1,keepdims=True)
            y_pred = alpha / tf.reduce_sum(
                alpha, axis=1, keepdims=True
            )  # (32, 5)
            # yをone-hot表現にして送る
            y = tf.one_hot(y, depth=self.n_class)  # (32, 5)
            # Loss
            loss = self.compiled_loss(
                y, alpha, regularization_losses=self.losses
            )

        # Gradients
        training_vars = self.trainable_variables
        gradients = tape.gradient(loss, training_vars)

        # Step with optimizer
        self.optimizer.apply_gradients(zip(gradients, training_vars))
        # accuracyのメトリクスにはy_predを入れる
        self.compiled_metrics.update_state(y, y_pred)
        # loss: edlのロス，accuracy: edlの出力が合っているか
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        evidence = self(x, training=False)
        alpha = evidence + 1
        y_pred = alpha / tf.reduce_sum(alpha, axis=1, keepdims=True)
        # uncertainty = self.n_class/tf.reduce_sum(alpha, axis=1,keepdims=True)
        # Updates the metrics tracking the loss
        # yをone-hot表現にして送る
        y = tf.one_hot(y, depth=self.n_class)
        # loss = self.compiled_loss(y, alpha)  # TODO : これいる？
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        metrics_dict = {m.name: m.result() for m in self.metrics}
        # metrics_dict.update(u_dict)
        return metrics_dict

    @tf.function
    def u_accuracy(self, y_true, y_pred, uncertainty, u_threshold=0):
        assert np.ndim(uncertainty) == 2  # (batch, 1)
        assert y_true.shape == y_pred.shape  # (batch, n_class)
        _y_true_list = list()
        _y_pred_list = list()
        for _y_true, _y_pred, _u in zip(y_true, y_pred, uncertainty):
            if _u > u_threshold:
                _y_true_list.append(_y_true)
                _y_pred_list.append(_y_pred)
        u_acc = tf.keras.metrics.categorical_accuracy(
            np.array(y_true), np.array(y_pred)
        )
        return {"u_acc": u_acc}


if __name__ == "__main__":
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # gpuのメモリエラーが起こる可能性があるのでcpuで計算
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import random

    random.seed(0)
    import numpy as np

    # edl_classifierのテストコード
    def edl_classifier_2d_checker():
        # 入力のサイズ
        batch_size = 10
        n_class = 5
        input_shape = (batch_size, 128, 512, 1)
        x = tf.random.normal(shape=input_shape)
        # 範囲はクラス数-1
        y = [random.randint(0, n_class - 1) for _ in range(batch_size)]
        y = np.array(y)

        # モデルの確認(edl_classifier_2d)
        # shapeはバッチサイズ以降の形を指定
        inputs = tf.keras.Input(shape=input_shape[1:])
        outputs = edl_classifier_2d(
            x=inputs, n_class=n_class, has_attention=True
        )
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # モデルのコンパイル
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=["accuracy"],
        )

        # コールバックのためにグラフを作成
        log_dir = (
            "logs/my_random_fit_graph/"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        # 訓練
        model.fit(
            x=x, y=y, epochs=5, callbacks=[tensorboard_callback], batch_size=4
        )
        return

    # TODO : model_baseのテストコード（使う必要ないので後回し）
    def model_base_checker():
        return

    # TODO : create_model_baseのテストコード（使う必要ないので後回し）
    def create_model_base_checker():
        return

    # TODO : edl_model_baseのテストコード
    def edl_model_base_checker():
        # 入力のサイズ
        batch_size = 10
        n_class = 5
        input_shape = (batch_size, 128, 512, 1)
        x = tf.random.normal(shape=input_shape)
        # 範囲はクラス数-1
        y = [random.randint(0, n_class - 1) for _ in range(batch_size)]
        y = np.array(y)

        # モデルの確認(edl_classifier_2d)
        # shapeはバッチサイズ以降の形を指定
        inputs = tf.keras.Input(shape=input_shape[1:])
        outputs = edl_classifier_2d(
            x=inputs, n_class=n_class, has_attention=True
        )
        model = EDLModelBase(inputs=inputs, outputs=outputs)

        # モデルのコンパイル
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=["accuracy"],
        )

        # コールバックのためにグラフを作成
        log_dir = (
            "logs/in_edl_model_base_checker/"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        # 訓練
        model.fit(
            x=x, y=y, epochs=50, callbacks=[tensorboard_callback], batch_size=4
        )
        return

    # edl_classifierのテストコード
    def edl_classifier_1d_checker():
        # 入力のサイズ
        batch_size = 10
        n_class = 5
        input_shape = (batch_size, 512, 1)
        x = tf.random.normal(shape=input_shape)
        # 範囲はクラス数-1
        y = [random.randint(0, n_class - 1) for _ in range(batch_size)]
        y = np.array(y)

        # モデルの確認(edl_classifier_2d)
        # shapeはバッチサイズ以降の形を指定
        inputs = tf.keras.Input(shape=input_shape[1:])
        outputs = edl_classifier_1d(
            x=inputs, n_class=n_class, has_attention=True
        )
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # モデルのコンパイル
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=["accuracy"],
        )

        # コールバックのためにグラフを作成
        log_dir = (
            "logs/my_random_fit_graph/"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        # 訓練
        model.fit(
            x=x, y=y, epochs=5, callbacks=[tensorboard_callback], batch_size=4
        )
        return

    # チェックしたい関数（クラス）のみTrueにする
    check_edl_classifier_2d = False
    check_model_base = False
    check_create_model_base = False
    check_edl_model_base = False
    check_edl_classifier_1d = True

    # edl_classifierのモデルをチェックしたいとき
    if check_edl_classifier_2d:
        edl_classifier_2d_checker()
    if check_model_base:
        model_base_checker()
    if check_create_model_base:
        create_model_base_checker()
    if check_edl_model_base:
        edl_model_base_checker()
    if check_edl_classifier_1d:
        edl_classifier_1d_checker()
