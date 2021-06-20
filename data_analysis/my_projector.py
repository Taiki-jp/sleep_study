from pre_process.load_sleep_data import LoadSleepData
from pre_process.pre_process import PreProcess
from data_analysis.py_color import PyColor
import sys, os
import tensorflow as tf
from tensorboard.plugins import projector
from nn.losses import EDLLoss

def main(train, 
         test,
         pre_process, 
         hidden_layer_id,
         test_name,
         data_type,
         pse_data=False,
         date_id=None):

    # データセットの作成
    (x_train, y_train), (x_test, y_test) = pre_process.make_dataset(train=train, 
                                                                    test=test, 
                                                                    is_storchastic=False,
                                                                    pse_data=pse_data,
                                                                    to_one_hot_vector=False,
                                                                    is_shuffle=False,
                                                                    each_data_size=100)
    # hidden_layer_idがNoneのときは入力空間を渡す
    if hidden_layer_id == "input":
        train_test_label = ["train", "test"]
        train_test_holder = [(x_train, y_train), (x_test, y_test)]
        for data, train_or_test in zip(train_test_holder, train_test_label):
            x, y = data
            log_dir = os.path.join(os.environ["sleep"],
                                   "logs",
                                   test_name,
                                   data_type,
                                   f"layer_{hidden_layer_id}",
                                   train_or_test)
            # ディレクトリの作成
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            # ラベルの保存
            label_path = os.path.join(log_dir, "metadata.tsv")
            with open(label_path, "w") as f:
                f.write("index\tlabel\n")
                for index, label in enumerate(y):
                    f.write(f"{index}\t{str(label)}\n")
            # チェックポイントの作成
            embedding_var = tf.Variable(x.reshape(-1, 512), name="hidden")  # spectrumのみの実装
            check_point_file = os.path.join(log_dir, "embedding.ckpt")
            ckpt = tf.train.Checkpoint(embedding=embedding_var) 
            ckpt.save(check_point_file)
            # projectorの設定
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
            embedding.metadata_path = "metadata.tsv"
            projector.visualize_embeddings(log_dir, config)
    else:
        # モデルの読み込み（コンパイル済み）
        print(f"{py_color.CYAN}{py_color.RETURN}*** {test_name}のモデルを読み込みます ***{py_color.END}")
        path = os.path.join(os.environ["sleep"], "models", test_name, date_id)
        model = tf.keras.models.load_model(path, custom_objects={"EDLLoss":EDLLoss(K=5,annealing=0.1)})
        print(f"*** {test_name}のモデルを読み込みました ***")
        # 新しいモデルの作成
        new_model = tf.keras.Model(inputs=model.input,
                                   outputs=model.layers[hidden_layer_id].output)


        # モデルの評価（どの関数が走る？ => lossのcallが呼ばれてる） 
        # NOTE : そのためone-hotの状態でデータを読み込む必要がある

        # trainとtestのループ処理
        train_test_holder = [(x_train, y_train), (x_test, y_test)]
        train_test_label = ["train", "test"]
        for train_or_test, data in zip(train_test_label, train_test_holder):
            x, y = data
            # EDLBase.__call__が走る
            hidden = new_model.predict(x, batch_size=32)
            # uncertaintyの計算
            evidence = model.predict(x, batch_size=32)
            alpha = evidence + 1
            uncertainty = 5/tf.reduce_sum(alpha, axis=1, keepdims=True)
            uncertainty = [_uncertainty.numpy()[0] for _uncertainty in uncertainty]
            # log先の設定
            log_dir = os.path.join(os.environ["sleep"],
                                   "logs",
                                   test_name,
                                   data_type,
                                   f"layer_{hidden_layer_id}",
                                   train_or_test)
            # ディレクトリの作成
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            # ラベルの保存
            label_path = os.path.join(log_dir, "metadata.tsv")
            with open(label_path, "w") as f:
                f.write("index\tlabel\tuncrt\n")
                for index, (label, u) in enumerate(zip(y, uncertainty)):
                    f.write(f"{index}\t{str(label)}\t{str(u)}\n")
            # チェックポイントの作成
            embedding_var = tf.Variable(hidden, name="hidden")
            check_point_file = os.path.join(log_dir, "embedding.ckpt")
            ckpt = tf.train.Checkpoint(embedding=embedding_var) 
            ckpt.save(check_point_file)
            # projectorの設定
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
            embedding.metadata_path = "metadata.tsv"
            projector.visualize_embeddings(log_dir, config)

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
    HIDDEN_LAYER_ID = "input"
    
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
    
    for test_id, (test_name, date_id) in enumerate(zip(load_sleep_data.sl.name_list, date_id_list)):
        (train, test) = pre_process.split_train_test_from_records(datasets,
                                                                  test_id=test_id, 
                                                                  pse_data=PSE_DATA)
        main(pre_process=pre_process,
             train=train,
             hidden_layer_id=HIDDEN_LAYER_ID,
             test_name=test_name,
             test=test,
             date_id=date_id, 
             pse_data=PSE_DATA,
             data_type=DATA_TYPE)
             
