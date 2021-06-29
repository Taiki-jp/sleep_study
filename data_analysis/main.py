from pre_process.subjects_list import SubjectsList
from data_analysis.utils import Utils
import os, datetime, wandb
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # tensorflow を読み込む前のタイミングですると効果あり
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from wandb.keras import WandbCallback
from pre_process.pre_process import PreProcess
from pre_process.load_sleep_data import LoadSleepData
from nn.losses import EDLLoss
import time, sys

def main(name, project, train, test,
         pre_process, load_model=False, my_tags=None, 
         pse_data=False, test_name=None, date_id=None,
         utils=None):

    # データセットの作成
    (x_train, y_train), (x_test, y_test) = pre_process.make_dataset(train=train, 
                                                                    test=test, 
                                                                    is_storchastic=False,
                                                                    pse_data=pse_data,
                                                                    to_one_hot_vector=False)
    # データセットの数
    print(f"training data : {x_train.shape}")

    # wandbの初期化
    wandb.init(name = name, 
               project = project,
               tags = my_tags,
               config= {"test name":test_name,
                        "date id":date_id},
               dir = utils.project_dir)
    
    # モデルの読み込み（コンパイル済み）
    if load_model:
        print(f"*** {test_name}のモデルを読み込みます ***")
        path = os.path.join(os.environ["sleep"], "models", test_name, date_id)
        model = tf.keras.models.load_model(path, custom_objects={"EDLLoss":EDLLoss(K=5,annealing=0.1)})
        print(f"*** {test_name}のモデルを読み込みました ***")

    
    # モデルの評価（どの関数が走る？ => lossのcallが呼ばれてる） 
    # NOTE : そのためone-hotの状態でデータを読み込む必要がある
    
    # trainとtestのループ処理
    train_test_holder = [(x_train, y_train), (x_test, y_test)]
    train_test_label = ["train", "test"]
    for train_or_test, data in zip(train_test_label, train_test_holder):
        x, y = data
        # EDLBase.__call__が走る
        evidence = model.predict(x, batch_size=32)
        # 混合行列をwandbに送信
        utils.conf_mat2Wandb(y=y, evidence=evidence, 
                             train_or_test=train_or_test,
                             test_label=test_name,
                             date_id=date_id)
        # 不確かさのヒストグラムをwandbに送信
        utils.u_hist2Wandb(y=y, 
                           evidence=evidence, 
                           train_or_test=train_or_test,
                           test_label=test_name,
                           date_id=date_id,
                           separate_each_ss=True)
        # 閾値を設定して分類した時の一致率とサンプル数をwandbに送信
        utils.u_threshold_and_acc2Wandb(y=y, evidence=evidence, 
                                        train_or_test=train_or_test,
                                        test_label=test_name, date_id=date_id)
        # 先にwandbが閉じないように10秒待つ
        # time.sleep(10)
    # wandb終了
    wandb.finish()

def load_date_id_list(has_attention, has_inception, data_type, lsd):
    # attention, inception, data_typeによって読み込むモデルを決める(8通り)
    # attention なしのものは現在存在しない
    if has_attention is not True:
        print("attentionはすべてTrueにしてください")
        sys.exit(1)
    elif has_attention:
        if has_inception:
            if data_type == "spectrogram":
                return lsd.sl.date_id_list_attnt_incpt_spc_2d
            elif data_type == "spectrum":
                return lsd.sl.date_id_list_attnt_incpt_spc_1d
            else:
                sys.exit(1)
        else:
            if data_type == "spectrogram":
                return lsd.sl.date_id_list_attnt_spc_2d
            elif data_type == "spectrum":
                return lsd.sl.date_id_list_attnt_spc_1d
            else:
                sys.exit(1)
                
if __name__ == '__main__':
    # 環境設定
    try:
        tf.keras.backend.set_floatx('float32')
        tf.config.run_functions_eagerly(True)
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        print("*** cpuで計算します ***")
    
    # ハイパーパラメータの設定
    (HAS_ATTENTION, PSE_DATA, HAS_INCEPTION, DATA_TYPE) = (True, False, True, "spectrogram")
    ATTENTION_TAG = "attention" if HAS_ATTENTION else "no-attention"
    PSE_DATA_TAG = "psedata" if PSE_DATA else "sleepdata"
    INCEPTION_TAG = "inception" if HAS_INCEPTION else "no-inception"
    WANDB_PROJECT = "test" if PSE_DATA else "edl-analysis"
    my_tags = [ATTENTION_TAG, 
               PSE_DATA_TAG,
               INCEPTION_TAG,
               DATA_TYPE]
    
    # オブジェクトの作成
    load_sleep_data = LoadSleepData(data_type=DATA_TYPE)
    pre_process = PreProcess(load_sleep_data)
    datasets = load_sleep_data.load_data(load_all=True, pse_data=PSE_DATA)
    utils = Utils(file_reader=load_sleep_data.fr)

    # 読み込むモデルの日付リストを返す
    date_id_list = load_date_id_list(has_attention=HAS_ATTENTION,
                                     has_inception=HAS_INCEPTION,
                                     data_type=DATA_TYPE,
                                     lsd=load_sleep_data)
    
    for test_id, (test_name, date_id) in enumerate(zip(load_sleep_data.sl.name_list, date_id_list)):
        (train, test) = pre_process.split_train_test_from_records(datasets,
                                                                  test_id=test_id, 
                                                                  pse_data=PSE_DATA)
        main(name = f"edl-{test_name}", project=WANDB_PROJECT,
             pre_process=pre_process,train=train, 
             test=test, load_model=True, 
             my_tags=my_tags+[f"{test_name}"],
             date_id=date_id, pse_data=PSE_DATA,
             test_name=test_name,utils=utils)
