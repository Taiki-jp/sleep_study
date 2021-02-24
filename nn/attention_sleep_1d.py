# ================================================ #
# *            ライブラリのインポート
# ================================================ #
from my_setting import SetsPath, FindsDir
SetsPath().set()
import datetime, wandb
# * データ保存用ライブラリ
from wandb.keras import WandbCallback
# * モデル計算初期化用ライブラリ
import tensorflow as tf
# * モデル構築ライブラリ
from my_model import MyConv1DModel as Model
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
#  *     データ拡張を行うためにジェネレータ作成
# ================================================ #

(x_train, x_test) = m_preProcess.makeSleepStageSpectrum()
(y_train, y_test) = m_preProcess.makeSleepStage()


# ================================================ #
#  *             モデル保存時のための変数
# ================================================ #

id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# ================================================ #
#  *             データ保存先の設定
# ================================================ #

wandb.init(name = f"sleep : {id}", project = "test")
callBacks = [WandbCallback(validation_data = (x_test, y_test))]

# ================================================ #
#*         モデル作成（ネットから取ってくる方）       
# ! output_layer = 17 を取ってくると一回平行の処理を行っている
# ================================================ #

m_model = Model(n_classes = 5, with_attention = True, findsDirObj=m_findsDir)

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
                  epochs = 10,
                  callbacks = callBacks,
                  verbose = 2)

# ================================================ #
#*                   モデルの保存
# ================================================ #

m_model.saveModel(id = id)