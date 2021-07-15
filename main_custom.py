from random import sample
import sys
from data_analysis.py_color import PyColor
import os
import datetime
import wandb
import tensorflow as tf
from collections import Counter
from pre_process.pre_process import PreProcess
from nn.model_base import edl_classifier_1d
from nn.losses import EDLLoss
from pre_process.json_base import JsonBase
import numpy as np
from data_analysis.utils import Utils
from pre_process.record import Record, multipleRecords

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # tensorflow を読み込む前のタイミングですると効果あり


def main(
    utils,
    name,
    project,
    train,
    test,
    pre_process,
    model_save=False,
    epochs=1,
    save_model=False,
    my_tags=None,
    batch_size=32,
    n_class=5,
    pse_data=False,
    test_name=None,
    date_id=None,
    has_attention=False,
    has_inception=False,
    data_type="",
    sample_size=0,
    wandb_config=dict(),
    kernel_size=0,
):

    # データセットの作成
    (x_train, y_train), (x_test, y_test) = pre_process.make_dataset(
        train=train,
        test=test,
        is_storchastic=False,
        pse_data=pse_data,
        each_data_size=sample_size,
    )
    # カテゴリカルに変換
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    # データセットの数
    print(f"training data : {x_train.shape}")
    ss_train_dict = Counter(y_train)
    ss_test_dict = Counter(y_train)
    # 不確かさのリストを作成（初期値0)
    u_train = multipleRecords(len(y_train))
    u_test = multipleRecords(len(y_test))

    # カスタムトレーニングのために作成
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, u_train))
    train_dataset = train_dataset.shuffle(buffer_size=x_train.shape[0]).batch(
        batch_size
    )
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(batch_size)

    # config の追加
    added_config = {
        "attention": has_attention,
        "inception": has_inception,
        "test wake before replaced": ss_test_dict[4],
        "test rem before replaced": ss_test_dict[3],
        "test nr1 before replaced": ss_test_dict[2],
        "test nr2 before replaced": ss_test_dict[1],
        "test nr34 before replaced": ss_test_dict[0],
        "train wake before replaced": ss_train_dict[4],
        "train rem before replaced": ss_train_dict[3],
        "train nr1 before replaced": ss_train_dict[2],
        "train nr2 before replaced": ss_train_dict[1],
        "train nr34 before replaced": ss_train_dict[0],
    }
    wandb_config = wandb_config.update(added_config)

    # wandbの初期化
    wandb.init(
        name=name,
        project=project,
        tags=my_tags,
        config=wandb_config,
        sync_tensorboard=True,
        dir=pre_process.my_env.project_dir,
    )

    # モデルの作成とコンパイル
    if data_type == "spectrum":
        shape = (int(kernel_size / 2), 1)
    elif data_type == "spectrogram":
        shape = (128, 512, 1)
    else:
        print("correct here based on your model")
        sys.exit(1)

    inputs = tf.keras.Input(shape=shape)
    outputs = edl_classifier_1d(
        x=inputs,
        n_class=n_class,
        has_attention=has_attention,
        has_inception=has_inception,
    )

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # 最適化関数の設定
    optimizer = tf.keras.optimizers.Adam()
    # メトリクスの作成
    # true side : カテゴリカルな状態，pred side : クラスの次元数（ソフトマックスをかける前）
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # エポックのループ
    for epoch in range(epochs):
        # ロス関数はエポックごとにアニーリングを変えるので中に書く
        loss_fn = EDLLoss(K=n_class, annealing=min(1, (epoch / epochs)))
        print(f"エポック:{epoch + 1}")
        # エポック内のバッチサイズごとのループ
        for step, (x_batch_train, y_batch_train, unc) in enumerate(train_dataset):
            # 勾配を計算
            with tf.GradientTape() as tape:
                # NOTE : x_batch_trainの次元は3である
                # assert np.ndim(x_batch_train) == 3
                # dataset.shuffleを入れることによってバッチサイズを設定できる
                evidence = model(x_batch_train, training=True)
                alpha = evidence + 1
                y_pred = alpha / tf.reduce_sum(alpha, axis=1, keepdims=True)
                # y_trueをone-hotに変換して渡す
                loss_value = loss_fn.call(
                    tf.keras.utils.to_categorical(
                        y_batch_train, num_classes=5
                    ),
                    alpha,
                )  # NOTE : one-hotの形で入れるべき？

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # one-hot表現にそろえる方法（無理）
            # train_acc_metric.update_state(tf.keras.utils.to_categorical(y_batch_train, num_classes=5), y_pred)
            # one-hot表現にそろえる方法（無理）
            # sparse-categoricalの説明をしっかり見ると
            # y_batch_trainn <= integer, y_pred <= prob と書いてある
            train_acc_metric.update_state(y_batch_train, y_pred)

            # if step % 200 == 0:
            #    print(f"step:{step}, loss:{float(loss_value)}")

        # エポックの終わりにメトリクスを表示する
        train_acc = train_acc_metric.result()
        print(f"訓練一致率：{train_acc:.2%}")
        # エポックの終わりに訓練メトリクスを初期化
        train_acc_metric.reset_states()

        # 訓練の終わりに検証用データ（今回はテストデータ）の性能を見る
        for x_batch_val, y_batch_val in val_dataset:
            val_evidence = model(x_batch_val, training=False)
            val_alpha = val_evidence + 1
            val_y_pred = val_alpha / tf.reduce_sum(
                val_alpha, axis=1, keepdims=True
            )
            val_acc_metric.update_state(
                y_batch_val, val_y_pred
            )  # NOTE : one-hotの形で入れるべき？
        # メトリクスを表示
        val_acc = val_acc_metric.result()
        print(f"テスト一致率：{val_acc:.2%}")
        # 初期化
        val_acc_metric.reset_states()

        # wandbにログを送る（TODO：pre, rec, f-mも送る)
        wandb.log({"train_acc": train_acc, "test_acc": val_acc})

        # 混合マトリクスの作成（各エポック毎）
        # cm_train = utils.make_confusion_matrix(
        #     y_true=np.argmax(y_batch_train, axis=1), y_pred=np.argmax(y_pred, axis=1)
        # )  # 訓練データ
        # cm_test = utils.make_confusion_matrix(
        #     y_true=np.argmax(y_batch_val, axis=1), y_pred=np.argmax(val_y_pred, axis=1)
        # )
        # utils.save_image2wandb(
        #     cm_train, to_wandb=True, fileName="cm_train"
        # )
        # utils.save_image2wandb(
        #     cm_test, to_wandb=True, fileName="cm_test"
        # )

    # モデルの保存
    path = os.path.join(pre_process.my_env.models_dir, test_name, date_id)
    model.save(path)
    # wandbへの記録終了
    wandb.finish()


if __name__ == "__main__":
    # 環境設定
    CALC_DEVICE = "gpu"
    # CALC_DEVICE = "cpu"
    DEVICE_ID = "0" if CALC_DEVICE == "gpu" else "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID
    if os.environ["CUDA_VISIBLE_DEVICES"] != "-1":
        tf.keras.backend.set_floatx("float32")
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # tf.config.run_functions_eagerly(True)
    else:
        print("*** cpuで計算します ***")

    # ハイパーパラメータの設定
    TEST_RUN = False
    HAS_ATTENTION = True
    PSE_DATA = False
    HAS_INCEPTION = True
    IS_PREVIOUS = False
    IS_NORMAL = True
    IS_ENN = False
    EPOCHS = 25
    BATCH_SIZE = 32
    N_CLASS = 5
    KERNEL_SIZE = 512
    STRIDE = 16
    SAMPLE_SIZE = 5000
    DATA_TYPE = "spectrum"
    FIT_POS = "middle"
    NORMAL_TAG = "normal" if IS_NORMAL else "sas"
    ATTENTION_TAG = "attention" if HAS_ATTENTION else "no-attention"
    PSE_DATA_TAG = "psedata" if PSE_DATA else "sleepdata"
    INCEPTION_TAG = "inception" if HAS_INCEPTION else "no-inception"
    WANDB_PROJECT = "test" if TEST_RUN else "master"
    ENN_TAG = "enn" if IS_ENN else "dnn"

    # オブジェクトの作成
    pre_process = PreProcess(
        data_type=DATA_TYPE,
        fit_pos=FIT_POS,
        verbose=0,
        kernel_size=KERNEL_SIZE,
        is_previous=IS_PREVIOUS,
        stride=STRIDE,
        is_normal=IS_NORMAL,
    )
    datasets = pre_process.load_sleep_data.load_data(
        load_all=True,
        pse_data=PSE_DATA,
    )
    utils = Utils()
    # 記録用のjsonファイルを読み込む
    JB = JsonBase("../nn/model_id.json")
    JB.load()
    # モデルのidを記録するためのリスト
    date_id_saving_list = list()

    for test_id, test_name in enumerate(pre_process.name_list):
        date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        date_id_saving_list.append(date_id)
        (train, test) = pre_process.split_train_test_from_records(
            datasets, test_id=test_id, pse_data=PSE_DATA
        )
        # tagの設定
        my_tags = [
            test_name,
            PSE_DATA_TAG,
            ATTENTION_TAG,
            INCEPTION_TAG,
            DATA_TYPE,
            FIT_POS,
            f"kernel_{KERNEL_SIZE}",
            f"stride_{STRIDE}",
            f"sample_{SAMPLE_SIZE}",
            ENN_TAG,
        ]

        wandb_config = {
            "test name": test_name,
            "date id": date_id,
            "sample_size": SAMPLE_SIZE,
            "epochs": EPOCHS,
            "kernel": KERNEL_SIZE,
            "stride": STRIDE,
            "fit_pos": FIT_POS,
            "batch_size": BATCH_SIZE,
            "n_class": N_CLASS,
        }

        main(
            utils=utils,
            data_type=DATA_TYPE,
            model_save=True,
            name=test_name,
            project=WANDB_PROJECT,
            pre_process=pre_process,
            train=train,
            test=test,
            epochs=EPOCHS,
            save_model=True,
            has_attention=HAS_ATTENTION,
            my_tags=my_tags,
            date_id=date_id,
            pse_data=PSE_DATA,
            test_name=test_name,
            kernel_size=KERNEL_SIZE,
            wandb_config=wandb_config,
            sample_size=SAMPLE_SIZE,
        )

        if TEST_RUN:
            break
    JB.dump(
        keys=[
            ENN_TAG,
            DATA_TYPE,
            FIT_POS,
            f"stride_{str(STRIDE)}",
            f"kernel_{str(KERNEL_SIZE)}",
        ],
        value=date_id_saving_list,
    )
