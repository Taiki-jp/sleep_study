# ================================================ #
# *         Import Some Libraries
# ================================================ #

import os, datetime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# ================================================ #
# *     Model を継承した自作のモデルクラス
# ?             あまり使えない
# ================================================ #

class ModelBase(tf.keras.Model):
    def __init__(self, findsDirObj=None):
        super().__init__()
        if findsDirObj is not None:
            self.findsDirObj = findsDirObj
        else:
            print("findsDirObj is not set...")
    
    def saveModel(self, id):
        """tensorflow を使って保存するメソッド
        """
        path = os.path.join(self.findsDirObj.returnFilePath(), "models", id)
        self.model.save(path)
        
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
                 findsDirObj,
                 base_model = None, 
                 exploit_input_layer = 0, 
                 exploit_output_layer = None):
        """モデル作成のベースクラス

        Args:
            load_file_path ([bool]): [継承先でここにパスを入れると指定したファイルパスのモデルを読み込む]
            base_model ([type], optional): [description]. Defaults to None.
            exploit_input_layer (int, optional): [description]. Defaults to 0.
            exploit_output_layer ([type], optional): [description]. Defaults to None.
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
        """tensorflow を使って保存するメソッド
        """
        path = os.path.join(self.findsDirObj.returnFilePath(), "models", id)
        self.model.save(path)
        
    def autoCompile(self, model):
        model.compile(optimizer = tf.keras.optimizers.Adam(),
                      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                      metrics = ["accuracy"])
        
    def utilizeLayer(self, input_layer = None, output_layer = None):
        #! input_layer はデフォルトで０なのでその時は引数を指定して
        #! if 文が無視されるけど構わない
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

class CustomFit(tf.keras.Model):
    def __init__(self, model):
        super(CustomFit, self).__init__()
        self.model = model
        self.metric = tf.keras.metrics.SparseCategoricalAccuracy(name = 'accuracy')

    def compile(self, optimizer, loss):
        super(CustomFit, self).compile()
        self.optimizer = optimizer
        self.loss = loss
    
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            # Caclulate predictions
            y_pred = self.model(x, training=True)

            # Loss
            loss = self.loss(y, y_pred)

        # Gradients
        training_vars = self.trainable_variables
        gradients = tape.gradient(loss, training_vars)

        # Step with optimizer
        self.optimizer.apply_gradients(zip(gradients, training_vars))
        self.acc_metric.update_state(y, y_pred)

        return {"loss": loss, "accuracy": self.acc_metric.result()}
    
    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_pred = self.model(x, training=False)

        # Updates the metrics tracking the loss
        loss = self.loss(y, y_pred)

        # Update the metrics.
        self.acc_metric.update_state(y, y_pred)
        return {"loss": loss, "accuracy": self.acc_metric.result()}
