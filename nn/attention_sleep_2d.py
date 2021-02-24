# ================================================ #
# *            ライブラリのインポート
# ================================================ #
from my_setting import SetsPath, FindsDir
SetsPath().set()
import datetime, wandb
# * データ保存用ライブラリ
from wandb.keras import WandbCallback
from wandb_classification_callback import WandbClassificationCallback
# * モデル計算初期化用ライブラリ
import tensorflow as tf
# * モデル構築ライブラリ
from my_model import MyInceptionAndAttention as Model
from model_base import CustomFit
# * 前処理ライブラリ
from utils import PreProcess

# ================================================ #
# *                  計算時の設定
# ================================================ #

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.keras.backend.set_floatx('float32')

# ================================================ #
# *          前処理のオブジェクト作成
# ================================================ #
m_findsDir = FindsDir("sleep")
m_preProcess = PreProcess(project=m_findsDir.returnDirName())

# ================================================ #
#  *                データ作成
# ================================================ #

(x_train, x_test) = m_preProcess.makeSleepStageSpectrum()
(y_train, y_test) = m_preProcess.makeSleepStage()
x_train = x_train.reshape(-1, 128, 512, 1)
x_test = x_test.reshape(-1, 128, 512, 1)
dataGen = m_preProcess.dataArg()
dataGen.fit(x_train)

# ================================================ #
#  *             モデル保存時のための変数
# ================================================ #

id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# ================================================ #
#  *             データ保存先の設定
# ================================================ #

wandb.init(name = f"sleep : {id}", project = "attention")
callBack = WandbClassificationCallback(validation_data = (x_test, y_test),
                                       log_confusion_matrix=True,
                                       labels=["wake", "rem", "nr1", "nr2", "nr34"])
# ================================================ #
#*         モデル作成（ネットから取ってくる方）       
# ================================================ #

m_model = Model(n_classes = 5,
                hight = 128,
                width = 512,
                channel = 1,
                findsDirObj=m_findsDir)

# ================================================ #
#*               モデルのコンパイル
# ================================================ #

m_model.model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"])

# ================================================ #
#*                   モデル学習
# ================================================ #

m_model.model.fit(dataGen.flow(x_train, y_train, batch_size=16),
                  validation_data = (x_test, y_test),
                  batch_size = 16,
                  epochs = 0,
                  callbacks = [callBack,],
                  verbose = 2)

# ================================================ #
#*                   モデルの保存
# ================================================ #

m_model.saveModel(id = id)

