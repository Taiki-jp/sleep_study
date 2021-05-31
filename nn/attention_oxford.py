# ================================================ #
# *            ライブラリのインポート
# ================================================ #

import os, sys, datetime, wandb
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# * wandb をオフラインで実行したいときに設定する
# os.environ['WANDB_MODE'] = 'dryrun'
os.environ["WANDB_SILENT"] = "true"
sys.path.append(os.environ["USERPROFILE"] + "/git/sleepstudy/pythonscript/data_analysis")
sys.path.append(os.environ["USERPROFILE"] + "/git/sleepstudy/pythonscript/nn")
# * データ保存用ライブラリ
from wandb.keras import WandbCallback
# * モデル計算初期化用ライブラリ
import tensorflow as tf
# * モデル構築ライブラリ
from my_model import MyInceptionAndAttentionWithoutSubClassing as Model
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

m_preProcess = PreProcess()

# ================================================ #
#  *     データ拡張を行うためにジェネレータ作成
# ================================================ #

trainGenerator, validationGenerator, testGenerator = m_preProcess.make_generator()

# ================================================ #
#  *             モデル保存時のための変数
# ================================================ #

id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# ================================================ #
#  *             データ保存先の設定
# ================================================ #

wandb.init(name = f"oxford_pet : {id}", project = "attention")
callBacks = [WandbCallback(generator = validationGenerator, save_weights_only = True)]

# ================================================ #
#*         モデル作成（ネットから取ってくる方）       
# ! output_layer = 17 を取ってくると一回平行の処理を行っている
# ================================================ #

model = Model(num_classes = 37,
              hight = 224,
              width = 224,
              channel = 3,
              file_path = None).createModel(output_layer = -1)

# ================================================ #
#* 特徴量抽出の部分は学習しないで，アテンションの部分とその先のみを学習する
# ================================================ #
"""
for layer in model.layers:
    layer.trainable = False
    
# for layer in model.layers[-4:]:
#     layer.trainable = True

# attention mask だけ学習
model.layers[-4].trainable = True

model.summary()
"""
# ================================================ #
#*       モデルのコンパイル（サブクラスなし）
# ================================================ #

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

# ================================================ #
#*                   モデル学習
# ================================================ #

model.fit(trainGenerator,
          validation_data = validationGenerator,
          epochs = 100,
          callbacks = callBacks,
          verbose = 2)

# ================================================ #
#*                   モデルの保存
# ================================================ #

# TODO : 保存したいフォルダ名を決める
model.save(f"")