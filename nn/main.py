# ================================================ #
# *            ライブラリのインポート
# ================================================ #

import os
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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

def main(name, project, sleep_stage, train, test, 
         epoch=1, isSaveModel=False, my_tags=None, mul_num=False, 
         checkpoint_path=None, is_attention=True, my_confusion_file_name=None,
         id = None):
    
    TARGET_SS = sleep_stage
    # テストに関しては1:1の割合でmakeDatasetは作ってしまうので無視
    (x_train, y_train), (x_test, y_test) = m_preProcess.makeDataSet(train=train, 
                                                                    test=test, 
                                                                    is_split=True, 
                                                                    target_ss=TARGET_SS, 
                                                                    is_storchastic=True,  # 
                                                                    is_multiply=False,  # trueの時は今の実装では400を返す
                                                                    mul_num=mul_num)  
    m_preProcess.maxNorm(x_train)
    m_preProcess.maxNorm(x_test)
    (x_train, y_train) = m_preProcess.catchNone(x_train, y_train)
    (x_test, y_test) = m_preProcess.catchNone(x_test, y_test)
    from collections import Counter
    ss_train_dict = Counter(y_train)
    ss_test_dict = Counter(y_test)
    # ターゲットの睡眠段階とそれ以外でラベルを分けるために必要
    y_train = m_preProcess.binClassChanger(y_data=y_train, target_ss=TARGET_SS)
    y_test = m_preProcess.binClassChanger(y_data=y_test, target_ss=TARGET_SS)

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
                    "true label":ss_list_for_wandb[TARGET_SS-1],
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
    
    m_model = MyInceptionAndAttention(n_classes=2, 
                                      hight=128, 
                                      width=512, 
                                      findsDirObj=m_findsDir,
                                      is_attention=is_attention)
    
    # ================================================ #
    #*       モデルのコンパイル（サブクラスなし）
    # ================================================ #
    
    m_model.model.compile(optimizer=tf.keras.optimizers.Adam(),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=["accuracy"])
    
    # ================================================ #
    #*                   モデル学習
    # ================================================ #
    if Counter(y_test)[1] != 0:
        w_callBack = WandbClassificationCallback(validation_data = (x_test, y_test),
                                               training_data=(x_train, y_train),
                                               log_confusion_matrix=True,
                                               labels=["non-target", "target"],
                                               show_my_confusion_title=True,
                                               my_confusion_title=ss_list_for_wandb[TARGET_SS-1],
                                               my_confusion_file_name=my_confusion_file_name)
        
        tf_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=False, verbose=3)
        
        m_model.model.fit(x_train,
                          y_train,
                          validation_data = (x_test, y_test),
                          epochs = epoch,
                          callbacks = [w_callBack, tf_callback],
                          verbose = 2)
    else:
        w_callBack = WandbClassificationCallback()
        tf_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=False, verbose=3)
        print("テストデータが用意できません．学習のみ行います")
        m_model.model.fit(x_train,
                          y_train,
                          epochs=epoch,
                          callbacks=[w_callBack, tf_callback],
                          verbose=2)
    
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
    PROJECT = "sleep"
    m_findsDir = FindsDir(PROJECT)
    m_preProcess = PreProcess(project=m_findsDir.returnDirName(), input_file_name=Utils().name_dict)    

    for name in Utils().name_list[:-2][::-1]:  # テストデータとなる被験者データに対してループ処理を行っている, TODO : メモリ管理が上手くできるようになるまでこのループは避ける
    #for name in Utils().name_list:
        (train, test) = m_preProcess.loadData(is_split=True, is_auto_loop=True, is_auto_loop_name=name)
        for sleep_stage in [5]: #range(1, 6):  # 睡眠段階に対してループ処理を行っている
            mulNum = 1
            is_attention = True
            if is_attention:
                attention_tag = "attention"
            else:
                attention_tag = "no-attention"
            id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            import os
            checkpointPath = os.path.join(os.environ["sleep"], "models", name, attention_tag)
            m_preProcess.checkPath(checkpointPath)  
            checkpointPath = os.path.join(checkpointPath, "ss_"+str(sleep_stage))
            m_preProcess.checkPath(checkpointPath)
            checkpointPath = os.path.join(checkpointPath, "cp-{epoch:04d}.ckpt")
            cm_file_name = os.path.join(os.environ["sleep"], "analysis", f"{name}")
            m_preProcess.checkPath(cm_file_name)
            cm_file_name = os.path.join(cm_file_name, f"confusion_matrix_{sleep_stage}_{id}.csv")
            main(name = name, project = "sleep", sleep_stage=sleep_stage,
                 train=train, test=test, epoch=15, isSaveModel=False, mul_num=mulNum,
                 my_tags=["f measure", "testそのまま", f"train:1:{mulNum}", attention_tag],
                 checkpoint_path=checkpointPath, is_attention = is_attention, 
                 my_confusion_file_name=cm_file_name, id = id)
        break  # 一人だけを実行したいのでここにbreakを入れる
