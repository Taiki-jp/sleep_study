# ================================================ #
# *           Import Some Libraries
# ================================================ #

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# ================================================ #
# *   A Example of Implement of my own layer
# ================================================ #

class MyDense(tf.keras.layers.Layer):
    def __init__(self, units, input_dim):
        super(MyDense, self).__init__()
        self.w = self.add_weight(
            name="w",
            shape=(input_dim, units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name="b", shape=(units,), initializer="zeros", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# ================================================ #
# *    Attention Layer
# TODO : 色々なアテンションが様々なマスクになるように distribution のエラーをどこかに設定する
# ================================================ #
   
class MyAttention2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters = filters,
                                           kernel_size = kernel_size,)
        self.dropout = tf.keras.layers.Dropout(0.2)
    
    def call(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = tf.keras.activations.sigmoid(x)
        return x

class MyAttention1D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.layers.Conv1D(filters = 1, kernel_size = 1)
        self.dropout = tf.keras.layers.Dropout(0.2)
        
    def call(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = tf.keras.activations.sigmoid(x)
        return x
    
