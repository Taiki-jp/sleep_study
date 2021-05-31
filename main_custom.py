import os, datetime, wandb
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # tensorflow を読み込む前のタイミングですると効果あり
import tensorflow as tf
from collections import Counter
import numpy as np
from pre_process.pre_process import PreProcess
from pre_process.load_sleep_data import LoadSleepData
from nn.model_base import edl_classifier_2d
from nn.losses import EDLLoss

def main(name, project, train, test,
         pre_process,epochs=1, save_model=False, my_tags=None, batch_size=32, 
         n_class=5, pse_data=False, test_name=None, date_id=None, has_attention=False):

    # データセットの作成
    (x_train, y_train), (x_test, y_test) = pre_process.make_dataset(train=train, 
                                                                    test=test, 
                                                                    is_storchastic=False,
                                                                    pse_data=pse_data)
    # データセットの数
    print(f"training data : {x_train.shape}")
    ss_train_dict = Counter(y_train)
    ss_test_dict = Counter(y_test)

    # カスタムトレーニングのために作成
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=x_train.shape[0]).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(batch_size)

    # wandbの初期化
    wandb.init(name = name, 
           project = project,
           tags = my_tags,
           config= {"test name":test_name,
                    "date id":date_id,
                    "test wake before replaced":ss_test_dict[5],
                    "test rem before replaced":ss_test_dict[4],
                    "test nr1 before replaced":ss_test_dict[3],
                    "test nr2 before replaced":ss_test_dict[2],
                    "test nr34 before replaced":ss_test_dict[1],
                    "train wake before replaced":ss_train_dict[5],
                    "train rem before replaced":ss_train_dict[4],
                    "train nr1 before replaced":ss_train_dict[3],
                    "train nr2 before replaced":ss_train_dict[2],
                    "train nr34 before replaced":ss_train_dict[1]})
    
    inputs = tf.keras.Input(shape=(128, 512, 1))
    outputs = edl_classifier_2d(x=inputs, n_class=n_class, has_attention=has_attention)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # 最適化関数の設定
    optimizer = tf.keras.optimizers.Adam()
    # メトリクスの作成
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
                    "date id":date_id,  # モデル読み込み時にフォルダ名を特定するために必要な情報
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
        loss_fn = EDLLoss(K=n_class, annealing=(1-epoch/epochs))
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
        print(f"訓練一致率：{train_acc:.2%}")
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
        print(f"テスト一致率：{val_acc:.2%}")
        # 初期化
        val_acc_metric.reset_states()

        # wandbにログを送る（TODO：pre, rec, f-mも送る)
        wandb.log({"train_acc" : train_acc, "test_acc" : val_acc})
        
        # 混合マトリクスの作成（各エポック毎）
        cm_train, _ = pre_process.make_confusion_matrix(y_true=y_batch_train, y_pred=y_pred)  # 訓練データ
        cm_test, _ = pre_process.make_confusion_matrix(y_true=y_batch_val, y_pred=val_y_pred)
        pre_process.save_image2wandb(cm_train, to_wandb=True, fileName="cm_train")
        pre_process.save_image2wandb(cm_test, to_wandb=True, fileName="cm_test")
        
        # 不確かさのヒストグラム（各エポック毎）
        # 学習の最後に
        
    # wandbへの記録終了
    wandb.finish()


if __name__ == '__main__':   
    # 環境設定
    tf.keras.backend.set_floatx('float32')
    # tf.config.run_functions_eagerly(True)
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    # ハイパーパラメータの設定
    MUL_NUM = 1
    has_attention = True
    attention_tag = "attention" if has_attention else "no-attention"
    pse_data = True
    
    # オブジェクトの作成
    load_sleep_data = LoadSleepData(data_type="spectrogram")
    pre_process = PreProcess(load_sleep_data)
    datasets = load_sleep_data.load_data(load_all=True, pse_data=pse_data)

    # ループの開始
    for test_id, test_name in enumerate(load_sleep_data.sl.name_list):
        date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        (train, test) = pre_process.split_train_test_from_records(datasets, test_name, pse_data=pse_data)

        main(name = "edl", project = "edl",pre_process=pre_process,train=train, 
             test=test,epoch=100, save_model=True, has_attention=has_attention,
             my_tags=[f"{test_name}", f"train:1:{MUL_NUM}", attention_tag],date_id=date_id,
             pse_data=pse_data,test_name=test_name)
