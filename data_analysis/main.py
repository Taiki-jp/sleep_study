from data_analysis.utils import Utils
import os, datetime, wandb
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # tensorflow を読み込む前のタイミングですると効果あり
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from wandb.keras import WandbCallback
from pre_process.pre_process import PreProcess
from pre_process.load_sleep_data import LoadSleepData
from nn.losses import EDLLoss
import time

def main(name, project, train, test,
         pre_process, load_model=False, my_tags=None, batch_size=32, 
         n_class=5, pse_data=False, test_name=None, date_id=None, has_attention=False):

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
                        "date id":date_id})
    
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
        evidence = model(x)
        # Utilsオブジェクト作成
        utils = Utils()
        # 混合行列をwandbに送信
        """
        utils.conf_mat2Wandb(y=y, evidence=evidence, 
                             train_or_test=train_or_test,
                             test_label=test_name,
                             date_id=date_id)
        """
        # 不確かさのヒストグラムをwandbに送信
        utils.u_hist2Wandb(y=y, evidence=evidence, 
                           train_or_test=train_or_test,
                           test_label=test_name,
                           date_id=date_id)
        # 閾値を設定して分類した時の一致率とサンプル数をwandbに送信
        utils.u_threshold_and_acc2Wandb(y=y, evidence=evidence, 
                                        train_or_test=train_or_test,
                                        test_label=test_name, date_id=date_id)
        # 先にwandbが閉じないように10秒待つ
        # time.sleep(10)
    # wandb終了
    wandb.finish()
    
if __name__ == '__main__':
    # 環境設定
    try:
        tf.keras.backend.set_floatx('float32')
        #tf.config.run_functions_eagerly(True)
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        print("*** cpuで計算します ***")
    
    # ハイパーパラメータの設定
    MUL_NUM = 1
    has_attention = True
    attention_tag = "attention" if has_attention else "no-attention"
    pse_data = True
    pse_data_tag = "psedata" if pse_data else "sleepdata"
    
    # オブジェクトの作成
    load_sleep_data = LoadSleepData(data_type="spectrogram")
    pre_process = PreProcess(load_sleep_data)
    datasets = load_sleep_data.load_data(load_all=True, pse_data=pse_data)

    
    # enn x inception
    data_id_list = ["20210601-051642",
                    "20210601-053406",
                    "20210602-120441",   #yamamoto : "20210601-055045",
                    "20210601-060739",
                    "20210601-062423",
                    "20210601-064055",
                    "20210601-065654",
                    "20210602-165042", #hiromoto : ""20210601-071330",
                    "20210601-073045"]
    
    
    """ enn x no inception
    data_id_list = ["20210604-070048",
                    "20210604-071219",
                    "20210604-072311",
                    "20210604-073420",
                    "20210604-074528",
                    "20210604-075638",
                    "20210604-080726",
                    "20210604-081841",
                    "20210604-083019"]
    """
    
    for test_id, (test_name, date_id) in enumerate(zip(load_sleep_data.sl.name_list, data_id_list)):
        (train, test) = pre_process.split_train_test_from_records(datasets, test_id=test_id, pse_data=pse_data)
        main(name = f"edl-{test_name}", project = "edl-analysis",pre_process=pre_process,train=train, 
             test=test, load_model=True, has_attention=has_attention,
             my_tags=[f"{test_name}", f"train:1:{MUL_NUM}", attention_tag, pse_data_tag],
             date_id=date_id, pse_data=pse_data,test_name=test_name)
