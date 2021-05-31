import os, datetime, wandb
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # tensorflow を読み込む前のタイミングですると効果あり
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from collections import Counter
import numpy as np
from pre_process.pre_process import PreProcess
from pre_process.load_sleep_data import LoadSleepData
from nn.model_base import EDLModelBase, edl_classifier_2d
from nn.losses import EDLLoss
from wandb.keras import WandbCallback

def main(name, project, train, test,
         pre_process,epochs=1, save_model=False, my_tags=None, batch_size=32, 
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
                        "date id":date_id},
               sync_tensorboard=True)
           
    # モデルの作成とコンパイル
    inputs = tf.keras.Input(shape=(128, 512, 1))
    outputs = edl_classifier_2d(x=inputs, n_class=n_class, has_attention=has_attention)
    model = EDLModelBase(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=EDLLoss(K=n_class, annealing=0.1),
                  metrics=["accuracy"])
    
    # tensorboard作成
    log_dir = f"logs/my_edl/{test_name}/"+date_id
    tf_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_test, y_test),
              epochs=epochs, callbacks=[tf_callback, WandbCallback()], verbose=2)
    
    if save_model:
        path = os.path.join(os.environ["sleep"], "models", test_name, date_id)
        model.save(path)
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
    pse_data = False
    pse_data_tag = "psedata" if pse_data else "sleepdata"
    epochs = 100
    
    # オブジェクトの作成
    load_sleep_data = LoadSleepData(data_type="spectrogram")
    pre_process = PreProcess(load_sleep_data)
    datasets = load_sleep_data.load_data(load_all=True, pse_data=pse_data)

    for test_id, test_name in enumerate(load_sleep_data.sl.name_list):
        date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        (train, test) = pre_process.split_train_test_from_records(datasets, test_id=test_id, pse_data=pse_data)

        main(name = f"edl-{test_name}", project = "edl",pre_process=pre_process,train=train, 
             test=test,epochs=epochs, save_model=True, has_attention=has_attention,
             my_tags=[f"{test_name}", f"train:1:{MUL_NUM}", attention_tag, pse_data_tag],
             date_id=date_id, pse_data=pse_data,test_name=test_name)
