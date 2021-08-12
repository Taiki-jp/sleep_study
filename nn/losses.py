import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.ops.numpy_ops.np_arrays import ndarray

# EDLのロス関数
class EDLLoss(tf.keras.losses.Loss):
    def __init__(self, K, annealing, name="custom_edl_loss", reduction="auto"):
        super().__init__(name=name, reduction=reduction)
        self.K = K
        self.annealing = annealing
        pass

    def call(self, y_true, evidence, unc=None):
        # y_trueはone-hotの状態
        alpha = evidence + 1  # (32:batch_size, 5:n_class)
        return self.loss_eq5(y_true=y_true, alpha=alpha, unc=unc)

    def KL(self, alpha):
        # 1行K列の全て値1のtensorflow.python.framework.ops.EagerTensorオブジェクト作成
        beta = tf.constant(np.ones((1, self.K)), dtype=tf.float32)  # (1, 5)
        # alphaの和を計算
        S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)  # (32, 1, 5)

        # KL情報量
        KL = (
            tf.reduce_sum(
                (alpha - beta)
                * (tf.math.digamma(alpha) - tf.math.digamma(S_alpha)),
                axis=1,
                keepdims=True,
            )
            + tf.math.lgamma(S_alpha)
            - tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True)
            + tf.reduce_sum(tf.math.lgamma(beta), axis=1, keepdims=True)
            - tf.math.lgamma(tf.reduce_sum(beta, axis=1, keepdims=True))
        )
        return KL

    # NOTE : y_trueがどのような形で入っているかに注意する
    # （実行は通るが、損失関数が訳分からなくなる）
    def loss_eq5(self, y_true: ndarray, alpha: Tensor, unc: Tensor):
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)  # (32, 1)
        # 2乗誤差
        L_err = tf.reduce_sum(
            (y_true - (alpha / S)) ** 2, axis=1, keepdims=True
        )  # (32, 1, 5)
        # ディリクレ分布の分散
        L_var = tf.reduce_sum(
            alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdims=True
        )  # (32, 1)
        # 損失関数:lambda * KL_div
        KL_reg = self.KL((alpha - 1) * (1 - y_true) + 1)
        if unc is not None:

            def step(unc: Tensor, unc_th: Tensor):
                return unc * tf.cast(unc >= unc_th, dtype=tf.float32)

            # return step(unc, 0.8) * (L_err + L_var + self.annealing * KL_reg)
            return unc * (L_err + L_var + self.annealing * KL_reg)
        else:
            return L_err + L_var + self.annealing * KL_reg

    def loss_eq4(self, y_true, alpha):
        return

    def loss_eq3(self, y_trur, alpha):
        return

    def get_config(self):
        config = super().get_config()
        config.update({"K": self.K, "annealing": self.annealing})
        return config


class MyLoss(tf.keras.losses.Loss):
    # NOTE : from_logits = Trueのみの実装
    def __init__(self, name="my_loss", axis=-1):
        super().__init__(name=name)
        self.axis = axis
        pass

    def call(self, y_true, y_pred):
        # reference : https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/losses.py#L687-L759
        # 上の実際のコードを基にソフトマックスクロスエントロピーを再現
        # y_true.shape = (batch_size), y_pred.shape = (batch_size, num_classes)
        y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        # y_true = ops.convert_to_tensor_v2_with_dispatch(y_true)
        # return K.sparse
        return K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True, axis=self.axis
        )


if __name__ == "__main__":
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    np.random.seed(0)
    y_true = np.random.randint(0, 4, 10)
    evidence = np.random.rand(10, 5).astype("float32")
    edl_loss = EDLLoss(K=5, annealing=0.1)
    loss = edl_loss.call(y_true=y_true, evidence=evidence)
    print("y_true : ", y_true)
    print("evidence", evidence)
    print(loss)
