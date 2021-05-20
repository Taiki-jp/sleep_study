# ================================================ #
# *         Import Some Libraries
# ================================================ #

import os, datetime
# TODO : この部分削除
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# ================================================ #
# *     Model を継承した自作のモデルクラス
# ?             あまり使えない
# ================================================ #

class ModelBase(tf.keras.Model):
    def __init__(self, findsDirObj=None):
        """[初期化メソッド]

        Args:
            findsDirObj ([type], optional): [モデル保存のパスを見つけるために必要]. Defaults to None.
        """
        super().__init__()
        self.findsDirObj = findsDirObj
        self.model = None
    # TODO : 2snake_case
    def saveModel(self, id):
        """パスを指定しなくていい分便利

        Args:
            id ([string]): [名前が被らないように日付を渡す]
        """
        path = os.path.join(self.findsDirObj.returnFilePath(), "models", id)
        self.model.save(path)
    # TODO : 2snake_case 
    def autoCompile(self):
        return self.compile(optimizer = tf.keras.optimizers.Adam(),
                            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                            metrics = ["accuracy"])

# ================================================ #
# *　Model に対して様々な自動操作を行うクラスをそろえた関数
# ================================================ #

class CreateModelBase(object):
    
    def __init__(self, 
                 load_file_path,
                 findsDirObj=None,
                 base_model = None, 
                 exploit_input_layer = 0, 
                 exploit_output_layer = None):
        """モデル作成のベースクラス

        Args:
            load_file_path ([bool]): [継承先でここにパスを入れると指定したファイルパスのモデルを読み込む]
            findsDirObj([object]): [保存パスを見つけるためのオブジェクト] Defaults to None
            base_model ([model], optional): [ベース構造（特徴抽出）に用いたいモデルを入れる]. Defaults to None.
            exploit_input_layer (int, optional): [ベース構造が指定されていないときは0]. Defaults to 0.
            exploit_output_layer ([int], optional): [ベース構造が指定されていないときはNone]. Defaults to None.
        """
        self.load_file_path = load_file_path
        self.findsDirObj = findsDirObj
        self.time_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.base_model = base_model
        self.exploit_input_layer = exploit_input_layer
        self.exploit_output_layer = exploit_output_layer
        self.model = None
        pass
    
    def saveModel(self, id):
        """パスを指定しなくていい分便利

        Args:
            id ([string]): [名前が被らないように日付を渡す]
        """
        path = os.path.join(self.findsDirObj.returnFilePath(), "models", id)
        self.model.save(path)
        
    def autoCompile(self):
        self.model.compile(optimizer = tf.keras.optimizers.Adam(),
                           loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                           metrics = ["accuracy"])
        
    def utilizeLayer(self, input_layer = None, output_layer = None):
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
    def __init__(self,
                 findsDirObj,
                 n_class):
        """初期化メソッド

        Args:
            findsDirObj([object]): [保存パスを見つけるためのオブジェクト] Defaults to None
        """
        super().__init__()
        self.findsDirObj = findsDirObj
        self.time_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.n_class = n_class

    def saveModel(self, id):
        """パスを指定しなくていい分便利

        Args:
            id ([string]): [名前が被らないように日付を渡す]
        """
        path = os.path.join(self.findsDirObj.returnFilePath(), "models", id)
        self.save(path)

    def compile(self, 
                optimizer, 
                loss, 
                metrics):
        super().compile()
        self.optimizer = optimizer
        self.loss = loss
        self.my_metrics = metrics
    
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            # Caclulate predictions
            evidence = self.model(x, training=True)
            alpha = evidence+1
            #uncertainty = self.n_class/tf.reduce_sum(alpha, axis=1,keepdims=True)
            y_pred = alpha/tf.reduce_sum(alpha, axis=1, keepdims=True)
            # Loss
            loss = self.loss(y, alpha)

        # Gradients
        training_vars = self.trainable_variables
        gradients = tape.gradient(loss, training_vars)

        # Step with optimizer
        self.optimizer.apply_gradients(zip(gradients, training_vars))
        # accuracyのメトリクスにはy_predを入れる
        self.acc_metric.update_state(y, y_pred)
        # loss: edlのロス，accuracy: edlの出力が合っているか
        return {"loss": loss, "accuracy": self.acc_metric.result()}
    
    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        evidence = self.model(x, training=False)
        alpha = evidence+1
        y_pred = alpha/tf.reduce_sum(alpha, axis=1, keepdims=True)
        # Updates the metrics tracking the loss
        loss = self.loss(y, alpha)
        # Update the metrics.
        self.acc_metric.update_state(y, y_pred)
        return {"loss": loss, "accuracy": self.acc_metric.result()}
