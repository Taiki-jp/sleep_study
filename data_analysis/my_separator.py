from pre_process.load_sleep_data import LoadSleepData
from pre_process.pre_process import PreProcess
from data_analysis.py_color import PyColor
import sys, os
import tensorflow as tf
from tensorboard.plugins import projector
from nn.losses import EDLLoss
import pickle

def sep_dataset_base_on_unc(train, 
                            test,
                            pre_process, 
                            test_name,
                            data_type,
                            pse_data=False,
                            date_id=None,
                            u_threthold=0.5):

    # データセットの作成
    (x_train, y_train), _ = pre_process.make_dataset(train=train, 
                                                     test=test, 
                                                     is_storchastic=False,
                                                     pse_data=pse_data,
                                                     to_one_hot_vector=False,
                                                     is_shuffle=False,
                                                     each_data_size=100)

    # モデルの読み込み（コンパイル済み）
    print(py_color.CYAN, py_color.RETURN,
          f"*** {test_name}のモデルを読み込みます ***",
          py_color.END)
    # モデルを読み込むパスの設定
    path = os.path.join(os.environ["sleep"], "models", test_name, date_id)
    model = tf.keras.models.load_model(path, custom_objects={"EDLLoss":EDLLoss(K=5,annealing=0.1)})
    print(py_color.CYAN, py_color.RETURN,
          f"*** {test_name}のモデルを読み込みました ***",
          py_color.END)
    
    # 不確かさの計算
    evidence = model.predict(x_train, batch_size=32)
    alpha = evidence + 1
    uncertainty = 5/tf.reduce_sum(alpha, axis=1, keepdims=True)
    uncertainty = [_uncertainty.numpy()[0] for _uncertainty in uncertainty]

    x_train_low = x_train[uncertainty < u_threthold]
    x_train_high = x_train[uncertainty > u_threthold]
    y_train_low = y_train[uncertainty > u_threthold]
    y_train_high = y_train[uncertainty < u_threthold]
    
    return (x_train_low, y_train_low), (x_train_high, y_train_high)

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
        # tf.config.run_functions_eagerly(True)
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        print("*** cpuで計算します ***")
    
    # ハイパーパラメータの設定
    (HAS_ATTENTION, PSE_DATA, HAS_INCEPTION, DATA_TYPE) = (True, False, True, "spectrum")
    HIDDEN_LAYER_ID = -5
    
    # オブジェクトの作成
    load_sleep_data = LoadSleepData(data_type=DATA_TYPE, verbose=1)
    pre_process = PreProcess(load_sleep_data)
    datasets = load_sleep_data.load_data(load_all=True, pse_data=PSE_DATA)
    py_color = PyColor()

    # 読み込むモデルの日付リストを返す
    date_id_list = load_date_id_list(has_attention=HAS_ATTENTION,
                                     has_inception=HAS_INCEPTION,
                                     data_type=DATA_TYPE,
                                     lsd=load_sleep_data)
    
    # 確保用のリスト
    x_low_list = list()
    x_high_list = list()
    y_low_list = list()
    y_high_list = list()

    for test_id, (test_name, date_id) in enumerate(zip(load_sleep_data.sl.name_list, date_id_list)):
        (train, test) = pre_process.split_train_test_from_records(datasets,
                                                                  test_id=test_id, 
                                                                  pse_data=PSE_DATA)
        (x_low, y_low), (x_high, y_high) = sep_dataset_base_on_unc(pre_process=pre_process,
                                                                   train=train,
                                                                   hidden_layer_id=HIDDEN_LAYER_ID,
                                                                   test_name=test_name,
                                                                   test=test,
                                                                   date_id=date_id, 
                                                                   pse_data=PSE_DATA,
                                                                   data_type=DATA_TYPE)
        x_low_list.append(x_low)
        x_high_list.append(x_high)
        y_low_list.append(y_low)
        y_high_list.append(y_high)
    
    
    # まとめてpickle
    data = (x_low_list, y_low_list), (x_high_list, y_high_list)
    root_path = os.path.join(os.environ["sleep"],
                             "datas",
                             "pre_processed_data",
                             "u_sept")
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    file_path = os.path.join(root_path, f"{DATA_TYPE}.sav")
    pickle.dump(data, open(file_path, "wb"))       
