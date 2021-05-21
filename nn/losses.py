import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K

class EDLLoss(tf.keras.losses.Loss):
    def __init__(self, K, global_step=0.5, annealing_step=0.5,
                 name='custom_edl_loss', batch_size=8, reduction='auto'):
        super().__init__(name=name, reduction=reduction)
        self.K = K
        self.global_step = global_step
        self.annealing_step = annealing_step
        self.batch_size = batch_size
        pass
    
    def call(self, y_true, evidence):  # TODO : alphaじゃなくて確率の出力を渡した方が良い？
        #print("alpha", alpha)
        alpha = evidence+1
        return self.loss_eq5(y_true=y_true, alpha=alpha)
    
    def KL(self, alpha):
        # 1行K列の全て値1のtensorflow.python.framework.ops.EagerTensorオブジェクト作成
        beta = tf.constant(np.ones((1,self.K)),
                           dtype=tf.float32)
        # alphaの和を計算
        S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)

        # KL情報量
        KL = tf.reduce_sum((alpha - beta)*(tf.math.digamma(alpha)-tf.math.digamma(S_alpha)),axis=1,keepdims=True) + \
             tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(alpha),axis=1,keepdims=True) + \
             tf.reduce_sum(tf.math.lgamma(beta),axis=1,keepdims=True) - tf.math.lgamma(tf.reduce_sum(beta,axis=1,keepdims=True))
        return KL
    
    def loss_eq5(self, y_true, alpha):
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        # 2乗誤差
        _batch_dim, _ = alpha.shape
        # FIXME : eager executionのときのみエラーが起こらない
        # y_true = tf.one_hot(tf.reshape(y_true, _batch_dim), depth=5)
        L_err = tf.reduce_sum((y_true-(alpha/S))**2, axis=1, keepdims=True)
        # ディリクレ分布の分散
        L_var = tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True)
        # 損失関数:lambda * KL_div
        KL_reg =  tf.minimum(1.0, tf.cast(self.global_step/self.annealing_step, tf.float32)) * self.KL((alpha - 1)*(1-y_true) + 1)
        return L_err + L_var + 0.05*KL_reg
    
class MyLoss(tf.keras.losses.Loss):
    # NOTE : from_logits = Trueのみの実装
    def __init__(self, name='my_loss', axis=-1):
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
        #return K.sparse
        return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True, axis=self.axis)