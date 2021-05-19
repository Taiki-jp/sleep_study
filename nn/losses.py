import tensorflow as tf
import numpy as np

class EDLLoss(tf.keras.losses.Loss):
    def __init__(self, K, global_step=0.5, annealing_step=0.5,
                 name='custom_edl_loss'):
        super().__init__(name=name)
        self.K = K
        self.global_step = global_step
        self.annealing_step = annealing_step
        pass
    
    def call(self, y_true, alpha):
        print("alpha", alpha)
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
        L_err = tf.reduce_sum((y_true-(alpha/S))**2, axis=1, keepdims=True)
        # ディリクレ分布の分散
        L_var = tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True)
        # 損失関数:lambda * KL_div
        KL_reg =  tf.minimum(1.0, tf.cast(self.global_step/self.annealing_step, tf.float32)) * self.KL((alpha - 1)*(1-y_true) + 1)
        return L_err + L_var + KL_reg