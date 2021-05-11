import tensorflow as tf
import numpy as np

class EDLLoss(tf.keras.losses.Loss):
    def __init__(self, name, alpha, K, global_step, annealing_step):
        super().__init__(name=name)
        y_pred = alpha
        self.K = K
        self.global_step = global_step
        self.annealing_step = annealing_step
        pass
    
    def call(self, y_true, y_pred):
        return self.loss_eq5(y_true=y_true, y_pred=y_pred)
    
    def KL(self, y_pred):
        beta=tf.constant(np.ones((1,self.K)),dtype=tf.float32)
        S_alpha = tf.reduce_sum(y_pred,axis=1,keepdims=True)

        KL = tf.reduce_sum((y_pred - beta)*(tf.math.digamma(y_pred)-tf.math.digamma(S_alpha)),axis=1,keepdims=True) + \
             tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(y_pred),axis=1,keepdims=True) + \
             tf.reduce_sum(tf.math.lgamma(beta),axis=1,keepdims=True) - tf.math.lgamma(tf.reduce_sum(beta,axis=1,keepdims=True))
        return KL
    
    def loss_eq5(self, y_true, y_pred):
        S = tf.reduce_sum(y_pred, axis=1, keepdims=True)
        # 損失関数:L_err, L_var
        L_err = tf.reduce_sum((y_true-(y_pred/S))**2, axis=1, keepdims=True)
        L_var = tf.reduce_sum(y_pred*(S-y_pred)/(S*S*(S+1)), axis=1, keepdims=True)
        # 損失関数:lambda * KL_div
        KL_reg =  tf.minimum(1.0, tf.cast(self.global_step/self.annealing_step, tf.float32)) * self.KL((y_pred - 1)*(1-y_true) + 1)
        return L_err + L_var + KL_reg
        