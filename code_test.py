from nn.model_base import EDLModelBase
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.keras.backend import shape
from nn.my_model import MyInceptionAndAttention
from pre_process.utils import FindsDir
from random import randint
import numpy as np
import os, sys
from nn.losses import EDLLoss

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# tf.config.run_functions_eagerly(True)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
try:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("GPUがサポートされていません")
    
fd = FindsDir("sleep")

batch_size = 100
input_shape = (512, 128, 1)
inputs = tf.keras.Input(input_shape)

input_shape = (100, 512, 128, 1)
x = tf.random.normal(shape=input_shape)
y = [randint(0, 4) for _ in range(batch_size)]
# NOTE : numpy配列で渡すこと
y = np.array(y)
# edllossを使うときはonehotで渡す
y = tf.one_hot(y, depth=5)

classifier = MyInceptionAndAttention(n_classes=5, 
                                hight=128, 
                                width=512, 
                                findsDirObj=fd,
                                is_attention=True)

model = EDLModelBase(classifier=classifier,
                     findsDirObj=fd,
                     n_class=5)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=EDLLoss(K=5),
              metrics=['accuracy'])

model.fit(x, y, epochs=10)
print(model.summary())

model.save("tmp_save")

reconstructed_model = tf.keras.models.load_model("tmp_save",
                                                 custom_objects={"EDLLoss":EDLLoss(K=5)})

print(reconstructed_model.summary())

#np.testing.assert_allclose(model.predict(x), 
#                           reconstructed_model.predict(x))