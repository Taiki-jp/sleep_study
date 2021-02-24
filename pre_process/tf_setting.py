
# *TODO : このファイル内で設定した tensorflow の条件をインポート先に適応するためには如何するべきか。

import tensorflow as tf

class MyTensorFlow(tf):
    def __init__(self):
        super().__init__()
        pass
    
    def set(self):
        tf.keras.backend.set_floatx('float32')
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.keras.backend.set_floatx('float32')
        
    def useVersion1(self):
        config = tf.compat.v1.ConfigProto()
        
"""
メモ
tensorflow の実行時の warning を消す環境変数  TF_CPP_MIN_LOG_LEVEL

０：全てのメッセージが出力される（デフォルト）。
１：INFOメッセージが出ない。
２：INFOとWARNINGが出ない。
３：INFOとWARNINGとERRORが出ない。

"""