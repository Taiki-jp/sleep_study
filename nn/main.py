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
from load_sleep_data import LoadSleepData
from utils import PreProcess, Utils
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
                                                                    is_set_data_size=True, 
                                                                    target_ss=TARGET_SS, 
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
    
    def KL(alpha, K):
        beta=tf.constant(np.ones((1,K)),dtype=tf.float32)
        S_alpha = tf.reduce_sum(alpha,axis=1,keepdims=True)

        KL = tf.reduce_sum((alpha - beta)*(tf.digamma(alpha)-tf.digamma(S_alpha)),axis=1,keepdims=True) + \
             tf.lgamma(S_alpha) - tf.reduce_sum(tf.lgamma(alpha),axis=1,keepdims=True) + \
             tf.reduce_sum(tf.lgamma(beta),axis=1,keepdims=True) - tf.lgamma(tf.reduce_sum(beta,axis=1,keepdims=True))
        return KL

    def loss_eq5(pred, alpha, K, global_step, annealing_step):
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        loglikelihood = tf.reduce_sum((pred-(alpha/S))**2, axis=1, keepdims=True) + tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True)
        KL_reg =  tf.minimum(1.0, tf.cast(global_step/annealing_step, tf.float32)) * KL((alpha - 1)*(1-pred) + 1 , K)
        return loglikelihood + KL_reg
        
    
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
                                               my_confusion_file_name=my_confusion_file_name,
                                               log_f_measure=True,calc_metrics=True)
        
        tf_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=False, verbose=3)
        
        m_model.model.fit(x_train,
                          y_train,
                          batch_size=16,
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
                          batch_size=16,
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
    m_preProcess = PreProcess(input_file_name=Utils().name_dict)
    m_loadSleepData = LoadSleepData(input_file_name="H_Li")  # TODO : input_file_nameで指定したファイル名はload_data_allを使う際はいらない
    MUL_NUM = 1
    is_attention = True
    if is_attention:
        attention_tag = "attention"
    else:
        attention_tag = "no-attention"
    # for name in Utils().name_list[:-2][::-1]:  # テストデータとなる被験者データに対してループ処理を行っている, TODO : メモリ管理が上手くできるようになるまでこのループは避ける
    datasets = m_loadSleepData.load_data_all()
    id_list = [i for i in range(10)]
    for i, name in zip(id_list[5:], Utils().name_list[5:]):
        (train, test) = m_preProcess.split_train_test_from_records(datasets, test_id=i)
        for sleep_stage in range(1, 6):  # 睡眠段階に対してループ処理を行っている

            id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            
            checkpointPath = os.path.join(os.environ["sleep"], "models", name, attention_tag, 
                                          "ss_"+str(sleep_stage), "cp-{epoch:04d}.ckpt")
            cm_file_name = os.path.join(os.environ["sleep"], "analysis", f"{name}", 
                                        f"confusion_matrix_{sleep_stage}_{id}.csv")
            m_preProcess.check_path_auto(checkpointPath)
            m_preProcess.check_path_auto(cm_file_name)
            
            main(name = name, project = "sleep", sleep_stage=sleep_stage,
                 train=train, test=test, epoch=15, isSaveModel=False, mul_num=MUL_NUM,
                 my_tags=["f measure", "testそのまま", f"train:1:{MUL_NUM}", attention_tag],
                 checkpoint_path=checkpointPath, is_attention = is_attention, 
                 my_confusion_file_name=cm_file_name, id=id)
