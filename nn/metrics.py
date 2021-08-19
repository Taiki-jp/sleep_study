import tensorflow as tf
from tensorflow.keras import backend as K

# keras(tensorflow)のメトリクスはstateful, statelessな書き方がある
# stateful : metricsを継承して実装，stateless : 関数として実装


# class CategoricalTruePositives(tf.keras.metrics.Metric):
#     def __init__(self, name="categorical_true_positives", target_class=0, batch_size=1, data_size=0, n_class=0, **kwargs):
#         super().__init__(name=name+"_"+str(target_class), **kwargs)
#         self.true_positives = self.add_weight(name="ctp", initializer="zeros")
#         self.target_class = target_class
#         self.batch_size = batch_size
#         self.data_size = data_size
#         self.n_class = n_class

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         # =================================
#         # 動作しているけど、よくわからない動作
#         # =================================

#         # y_pred は one-hot表現, y_true は categorical 表現で来る？
#         y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
#         row, col = y_pred.shape
#         # y_pred の行数作成
#         _target_class = tf.reshape(tf.constant([self.target_class for _ in range(row)], dtype=tf.int32), shape=(row,1))
#         y_pred_values = tf.cast(y_pred, "int32") == _target_class
#         y_true_values = tf.cast(y_true, "int32") == _target_class
#         values = tf.cast(y_pred_values, "float32") * tf.cast(y_true_values, "float32")
#         # if sample_weight is not None:
#         #     sample_weight = tf.cast(sample_weight, "float32")
#         #     values = tf.multiply(values, sample_weight)
#         self.true_positives.assign_add(tf.reduce_sum(values))

#     def result(self):
#         return self.true_positives / self.batch_size / self.data_size

#     def reset_states(self):
#         # The state of the metric will be reset at the start of each epoch.
#         self.true_positives.assign(0.0)


class CategoricalTruePositives(tf.keras.metrics.Metric):
    def __init__(self, name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
        values = tf.cast(values, "float32")
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.0)


class RecallMetric(tf.keras.metrics.Metric):
    """ステートフルに Recall を計算するクラス"""

    def __init__(self, name="custom_recall", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        # 状態を貯めておく変数を用意する
        self.true_positives = tf.Variable(0.0)
        self.total_positives = tf.Variable(0.0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """新しく正解と予測が追加で与えられたときの処理"""
        true_positives = K.sum(y_true * y_pred)
        total_positives = K.sum(y_true)

        self.true_positives.assign_add(true_positives)
        self.total_positives.assign_add(total_positives)

    def result(self):
        """現時点の状態から計算されるメトリックを返す"""
        return self.true_positives / (self.total_positives + K.epsilon())

    def reset_states(self):
        """状態をリセットするときに呼ばれるコールバック"""
        self.true_positives.assign(0.0)
        self.total_positives.assign(0.0)


class AccuracyMetric(tf.keras.metrics.Metric):
    """ステートフルに Accuracy を計算するクラス"""

    def __init__(self, name="custom_accuracy", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

        self.matched = tf.Variable(0.0)
        self.unmatched = tf.Variable(0.0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_argmax = K.argmax(y_true)
        y_pred_argmax = K.argmax(y_pred)

        y_matched = K.sum(
            K.cast(K.equal(y_true_argmax, y_pred_argmax), dtype="float32")
        )
        y_unmatched = K.sum(
            K.cast(K.not_equal(y_true_argmax, y_pred_argmax), dtype="float32")
        )

        self.matched.assign_add(y_matched)
        self.unmatched.assign_add(y_unmatched)

    def result(self):
        return self.matched / (self.matched + self.unmatched)

    def reset_states(self):
        self.matched.assign(0.0)
        self.unmatched.assign(0.0)


# edlのためのメトリクス
class EDLMetrics(tf.keras.metrics.Metric):
    def __init__(self, name, dtype, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


class MyMetrics(tf.keras.metrics.Metric):
    def __init__(self, name="my_metrics", **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred):
        pass

    def result(self):
        pass

    def reset_states(self):
        pass


# 呼び方の確認のための簡単なステートレスメトリクス
def custom_metric(y_true, y_pred):
    return 1


if __name__ == "__main__":
    # データの作成
    # tf.config.run_functions_eagerly(True)
    from data_analysis.utils import Utils
    import numpy as np

    utils = Utils()
    (x_train, x_test), (y_train, y_test) = utils.point_symmetry_data(
        100, 2, 0, 0, 0
    )
    # y_trainのtypeがリストのままではだめ
    if type(y_train) == list:
        print("y_train の方をndarrayに変換します")
        y_train = np.array(y_train)
    # wandb に送る
    import wandb
    from wandb.keras import WandbCallback
    import os

    wandb.init(name="metric_test", project="test", dir=os.environ["sleep"])

    # モデルの作成
    import tensorflow as tf

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, activation="relu", input_shape=(2,)))
    model.add(tf.keras.layers.Dense(10, activation="relu"))
    model.add(tf.keras.layers.Dense(2, activation="relu"))
    print(model.summary())
    # モデルのコンパイル
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            "accuracy",
            custom_metric,
            CategoricalTruePositives(target_class=0, data_size=200, n_class=2),
            CategoricalTruePositives(target_class=1, data_size=200, n_class=2),
        ],
    )
    model.fit(
        x_train,
        y_train,
        batch_size=10,
        epochs=10,
        verbose=2,
        callbacks=[WandbCallback()],
    )
