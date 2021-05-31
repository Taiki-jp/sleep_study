# ================================================ #
# *            ライブラリのインポート
# ================================================ #
from my_setting import SetsPath, FindsDir
SetsPath().set()
import datetime, wandb
# * データ保存用ライブラリ
#from wandb.keras import WandbCallback
from wandb_classification_callback import WandbClassificationCallback
# * モデル計算初期化用ライブラリ
import tensorflow as tf
# * モデル構築ライブラリ
from my_model import MyInceptionAndAttention
# * 前処理ライブラリ
from utils import PreProcess, Utils

# ================================================ #
# *                  計算時の設定
# ================================================ #

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#tf.keras.backend.set_floatx('float32')

# ================================================ #
# *          前処理のオブジェクト作成
# ================================================ #

m_findsDir = FindsDir("sleep")
#inputFileName = input("*** 被験者名を入れてください *** \n")
m_preProcess = PreProcess(project=m_findsDir.returnDirName(), 
                          input_file_name=Utils().name_dict)

# ================================================ #
#  *     データ拡張を行うためにジェネレータ作成
# ================================================ #

(x_train, y_train), (x_test, y_test) = m_preProcess.makeDataSet(is_split=True)
m_preProcess.maxNorm(x_train)
m_preProcess.maxNorm(x_test)
(x_train, y_train) = m_preProcess.catchNone(x_train, y_train)
(x_test, y_test) = m_preProcess.catchNone(x_test, y_test)
y_train = m_preProcess.changeLabel(y_train)  # Counter({2: 346, 1: 2975, 0: 159, 3: 1105, 4: 458})
y_test = m_preProcess.changeLabel(y_test)  # Counter({2: 49, 1: 365, 4: 41, 0: 22, 3: 79})

# ================================================ #
#  *             モデル保存時のための変数
# ================================================ #

id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# ================================================ #
#  *             データ保存先の設定
# ================================================ #

wandb.init(name = f"sleep : {id}", project = "attention learning")
callBack = WandbClassificationCallback(validation_data = (x_test, y_test),
                                       log_confusion_matrix=True,
                                       labels=["nr34", "nr2", "nr1", "rem", "wake"])

# ================================================ #
#*         モデル作成（ネットから取ってくる方）       
# ! output_layer = 17 を取ってくると一回平行の処理を行っている
# ================================================ #

m_model = MyInceptionAndAttention(5, 128, 512, m_findsDir)

# ================================================ #
#*       モデルのコンパイル（サブクラスなし）
# ================================================ #

m_model.model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"])

# ================================================ #
#*                   モデル学習
# ================================================ #

m_model.model.fit(x_train,
                  y_train,
                  validation_data = (x_test, y_test),
                  epochs = 50,
                  callbacks = [callBack],
                  verbose = 2)

# ================================================ #
#*                   モデルの保存
# ================================================ #

m_model.saveModel(id = id)