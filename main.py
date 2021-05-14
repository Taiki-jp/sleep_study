# ================================================ #
# *            ライブラリのインポート
# ================================================ #

import os
from nn.my_setting import SetsPath, FindsDir
SetsPath().set()
import datetime, wandb
# * データ保存用ライブラリ
#from wandb.keras import WandbCallback
from pre_process.wandb_classification_callback import WandbClassificationCallback
# * モデル計算初期化用ライブラリ
import tensorflow as tf
# * モデル構築ライブラリ
from nn.my_model import MyInceptionAndAttention
# * 前処理ライブラリ
import numpy as np
from losses import EDLLoss
from pre_process.load_sleep_data import LoadSleepData
from pre_process.utils import PreProcess, Utils
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# NOTE : gpuを設定していない環境のためにエラーハンドル
try:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("GPUがサポートされていません")
tf.keras.backend.set_floatx('float32')

# ================================================ #
#  *                メイン関数
# ================================================ #

def main(name, project, train, test, 
         epoch=1, isSaveModel=False, my_tags=None, mul_num=False, 
         checkpoint_path=None, is_attention=True, my_confusion_file_name=None,
         id = None):
    
    # テストに関しては1:1の割合でmakeDatasetは作ってしまうので無視
    (x_train, y_train), (x_test, y_test) = m_preProcess.makeDataSet(train=train, 
                                                                    test=test, 
                                                                    is_set_data_size=True,
                                                                    mul_num=mul_num,
                                                                    is_storchastic=False) 
    m_preProcess.maxNorm(x_train)
    m_preProcess.maxNorm(x_test)
    (x_train, y_train) = m_preProcess.catchNone(x_train, y_train)
    (x_test, y_test) = m_preProcess.catchNone(x_test, y_test)
    from collections import Counter
    ss_train_dict = Counter(y_train)
    ss_test_dict = Counter(y_test)
    # convert label 1-5 to 0-4
    y_train-=1
    y_test-=1
    # ================================================ #
    #  *             データ保存先の設定
    # ================================================ #
    ss_list_for_wandb = ["NR34", "NR2", "NR1", "REM", "WAKE"]
    wandb.init(name = name, 
               project = project,
               tags = my_tags,
               config= \
               {
                   "test id":m_preProcess.test_data_for_wandb,
                    "train num":x_train.shape[0],
                    "test num":x_test.shape[0],
                    "date id":id,
                    "test wake before replaced":ss_test_dict[5],
                    "test rem before replaced":ss_test_dict[4],
                    "test nr1 before replaced":ss_test_dict[3],
                    "test nr2 before replaced":ss_test_dict[2],
                    "test nr34 before replaced":ss_test_dict[1],
                    "train wake before replaced":ss_train_dict[5],
                    "train rem before replaced":ss_train_dict[4],
                    "train nr1 before replaced":ss_train_dict[3],
                    "train nr2 before replaced":ss_train_dict[2],
                    "train nr34 before replaced":ss_train_dict[1]
                })
    
    # ================================================ #
    #*         モデル作成（ネットから取ってくる方）
    # ================================================ #
    
    m_model = MyInceptionAndAttention(n_classes=5, 
                                      hight=128, 
                                      width=512, 
                                      findsDirObj=m_findsDir,
                                      is_attention=is_attention)
    
    # ================================================ #
    #*       モデルのコンパイル（サブクラスなし）
    # ================================================ #
    
    m_model.model.compile(optimizer=tf.keras.optimizers.Adam(),
                          loss=EDLLoss(),
                          metrics=["accuracy"])
    
    # ================================================ #
    #*                   モデル学習
    # ================================================ #
    w_callBack = WandbClassificationCallback(validation_data = (x_test, y_test),
                                             training_data=(x_train, y_train),
                                             log_confusion_matrix=True,
                                             labels=["nr34", "nr2", "nr1", "rem", "wake"],
                                             show_my_confusion_title=True,
                                             my_confusion_title="confusion matrix",
                                             my_confusion_file_name=my_confusion_file_name,
                                             log_f_measure=False,calc_metrics=False)
    
    # NOTE : モデルの各チェックポイントでの保存部分は削除
    if checkpoint_path:
        pass
        # tf_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=False, verbose=3)
    
    m_model.model.fit(x_train,
                      y_train,
                      batch_size=64,
                      validation_data = (x_test, y_test),
                      epochs = epoch,
                      callbacks = [w_callBack],
                      verbose = 2)
    
    # ================================================ #
    #*                   モデルの保存
    # ================================================ #
    if isSaveModel:
        m_model.saveModel(id = id)
    wandb.finish()

# ================================================ #
#  *                ループ処理
# ================================================ #

if __name__ == '__main__':   

    m_findsDir = FindsDir("sleep")
    m_preProcess = PreProcess(input_file_name=Utils().name_dict)
    m_loadSleepData = LoadSleepData(input_file_name="H_Li")  # TODO : input_file_nameで指定したファイル名はload_data_allを使う際はいらない
    MUL_NUM = 1
    is_attention = True
    attention_tag = "attention" if is_attention else "no-attention"
    datasets = m_loadSleepData.load_data_all()
    # TODO : test-idと名前を紐づける
    for test_id in range(9):
        (train, test) = m_preProcess.split_train_test_from_records(datasets, test_id=test_id)
        id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
        #checkpointPath = os.path.join(os.environ["sleep"], "models", name, attention_tag, 
        #                              "ss_"+str(sleep_stage), "cp-{epoch:04d}.ckpt")
        cm_file_name = os.path.join(os.environ["sleep"], "analysis", f"{test_id}", f"confusion_matrix_{id}.csv")
        m_preProcess.check_path_auto(cm_file_name)
    
        main(name = "test", project = "test",
             train=train, test=test, epoch=15, isSaveModel=True, mul_num=MUL_NUM,
             my_tags=["f measure", "testそのまま", f"train:1:{MUL_NUM}", attention_tag],
             checkpoint_path=None, is_attention = is_attention, 
             my_confusion_file_name=cm_file_name, id=id)
