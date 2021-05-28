# ================================================ #
# *            ライブラリのインポート
# ================================================ #

from nn.model_base import EDLModelBase, edl_classifier_2d
import os
from nn.my_setting import SetsPath, FindsDir
SetsPath().set()
import datetime, wandb
from pre_process.wandb_classification_callback import WandbClassificationCallback
import tensorflow as tf
from nn.losses import EDLLoss, MyLoss
from pre_process.load_sleep_data import LoadSleepData
from pre_process.utils import PreProcess, Utils
from collections import Counter
import numpy as np

# NOTE : gpuを設定していない環境のためにエラーハンドル
try:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("GPUがサポートされていません")

# float32が推奨されているみたい
tf.keras.backend.set_floatx('float32')
# tf.functionのせいでデバッグがしずらい問題を解決してくれる（これを使わないことでエラーが起こらなかったりする）
#tf.config.run_functions_eagerly(True)
# ================================================ #
#  *                メイン関数
# ================================================ #

def main(name, project, train, test, epochs=1, my_tags=None, mul_num=False, 
         checkpoint_path=None, is_attention=True, my_confusion_file_name=None,
         id = None, batch_size=32, n_class=5):
    
    (x_train, y_train), (x_test, y_test) = m_preProcess.makeDataSet(train=train,
                                                                    test=test,
                                                                    is_set_data_size=True,
                                                                    mul_num=mul_num,
                                                                    is_storchastic=False)
    m_preProcess.maxNorm(x_train)
    m_preProcess.maxNorm(x_test)
    (x_train, y_train) = m_preProcess.catchNone(x_train, y_train)
    (x_test, y_test) = m_preProcess.catchNone(x_test, y_test)
    # input shape
    print(f"training data : {x_train.shape}")
    ss_train_dict = Counter(y_train)
    ss_test_dict = Counter(y_test)
    # convert label 1-5 to 0-4
    y_train-=1
    y_test-=1
    # convert2one-hot
    #y_train = tf.one_hot(y_train, n_class)
    #y_test = tf.one_hot(y_test, n_class)
    # change x shape
    x_train = x_train[:,:,:,np.newaxis]
    x_test = x_test[:,:,:,np.newaxis]
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(batch_size)
    
    inputs = tf.keras.Input(shape=(128, 512, 1))
    outputs = edl_classifier_2d(x=inputs, n_class=n_class, has_attention=has_attention)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam()
    # TODO : どのような形で引数に渡すべきかmnistの例をもとに調査する
    # true side : カテゴリカルな状態，pred side : クラスの次元数（ソフトマックスをかける前）
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    # wandbの初期化
    wandb.init(name = name, 
               project = project,
               tags = my_tags,
               config= \
               {
                   "train num":x_train.shape[0],
                    "test num":x_test.shape[0],
                    "date id":id,  # モデル読み込み時にフォルダ名を特定するために必要な情報
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
    
    # エポックのループ
    for epoch in range(epochs):
        # ロス関数はエポックごとにアニーリングを変えるので中に書く
        loss_fn = EDLLoss(K=n_class, annealing=epoch/epochs)
        print(f"エポック:{epoch}")
        # エポック内のバッチサイズごとのループ
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            # 勾配を計算
            with tf.GradientTape() as tape:
                # NOTE : x_batch_trainの次元は3である
                #assert np.ndim(x_batch_train) == 3
                # dataset.shuffleを入れることによってバッチサイズを設定できる
                evidence = model(x_batch_train, training=True)
                alpha = evidence+1
                y_pred = alpha/tf.reduce_sum(alpha, axis=1, keepdims=True)
                loss_value = loss_fn.call(y_batch_train, alpha)  # NOTE : one-hotの形で入れるべき？
            
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            
            train_acc_metric.update_state(y_batch_train, y_pred)
            
            #if step % 200 == 0:
            #    print(f"step:{step}, loss:{float(loss_value)}")
        
        # エポックの終わりにメトリクスを表示する
        train_acc = train_acc_metric.result()
        print(f"エポック中の訓練一致率：{train_acc:.2%}")
        # エポックの終わりに訓練メトリクスを初期化
        train_acc_metric.reset_states()
        
        
        # 訓練の終わりに検証用データ（今回はテストデータ）の性能を見る
        for x_batch_val, y_batch_val in val_dataset:
            val_evidence = model(x_batch_val, training=False)
            val_alpha = val_evidence+1
            val_y_pred = val_alpha/tf.reduce_sum(val_alpha, axis=1, keepdims=True)
            val_acc_metric.update_state(y_batch_val, val_y_pred)  # NOTE : one-hotの形で入れるべき？
        # メトリクスを表示
        val_acc = val_acc_metric.result()
        print(f"検証用データの一致率：{val_acc:.2%}")
        # 初期化
        val_acc_metric.reset_states()

        # wandbにログを送る
        wandb.log({"train_acc" : train_acc, "test_acc" : val_acc})
        
        # 混合マトリクスの作成
        
        
    # wandbへの記録終了
    wandb.finish()

# ================================================ #
#  *                ループ処理
# ================================================ #

if __name__ == '__main__':   

    fd = FindsDir("sleep")
    m_preProcess = PreProcess(input_file_name=Utils().name_dict)
    m_loadSleepData = LoadSleepData(input_file_name="H_Li")  # TODO : input_file_nameで指定したファイル名はload_data_allを使う際はいらない
    MUL_NUM = 1
    has_attention = True
    attention_tag = "attention" if has_attention else "no-attention"
    datasets = m_loadSleepData.load_data_all()
    # TODO : test-idと名前を紐づける
    for test_id in range(9):
        (train, test) = m_preProcess.split_train_test_from_records(datasets, test_id=test_id)
        id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
        #checkpointPath = os.path.join(os.environ["sleep"], "models", name, attention_tag, 
        #                              "ss_"+str(sleep_stage), "cp-{epoch:04d}.ckpt")
        cm_file_name = os.path.join(os.environ["sleep"], "analysis", f"{test_id}", f"confusion_matrix_{id}.csv")
        m_preProcess.check_path_auto(cm_file_name)
    
        main(name = list(Utils().name_dict.keys())[test_id], project = "edl",
             train=train, test=test, epochs=10, mul_num=MUL_NUM,
             my_tags=["loss_all", f"train:1:{MUL_NUM}", attention_tag],
             checkpoint_path=None, has_attention = has_attention, 
             my_confusion_file_name=cm_file_name, id=id)
