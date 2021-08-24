import copy
from random import sample
import sys
from data_analysis.py_color import PyColor
import os
import datetime
import wandb
import tensorflow as tf
from collections import Counter
from pre_process.pre_process import PreProcess
from nn.model_base import classifier4enn, spectrum_conv
from nn.losses import EDLLoss
from pre_process.json_base import JsonBase
import numpy as np
from data_analysis.utils import Utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # tensorflow を読み込む前のタイミングですると効果あり


def main(
    name: str,
    utils: Utils,
    project: str,
    train: list,
    test: list,
    pre_process: PreProcess,
    annealing_param: float,
    epochs: int = 1,
    my_tags: list = None,
    batch_size: int = 32,
    n_class: int = 5,
    pse_data: bool = False,
    test_name: str = None,
    date_id: str = None,
    has_attention: bool = False,
    has_inception: bool = False,
    data_type: str = "",
    sample_size: int = 0,
    wandb_config: dict = dict(),
    kernel_size: int = 0,
    is_mul_layer: bool = False,
    has_dropout: bool = False,
    subnet_starting_point: float = 0,
):

    # データセットの作成(one-hot で処理を行う)
    (x_train, y_train), (x_test, y_test) = pre_process.make_dataset(
        train=train,
        test=test,
        is_storchastic=False,
        pse_data=pse_data,
        each_data_size=sample_size,
        to_one_hot_vector=False,
    )
    # カテゴリカルに変換 TODO: make_dataset 内で onehot 表現に変えてもよいかチェック
    # データセットの数
    print(f"training data : {x_train.shape}")
    ss_train_dict = Counter(y_train)
    ss_test_dict = Counter(y_test)

    # カスタムトレーニングのために作成
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
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
    wandb_config.update(added_config)

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
    # NOTE: kernel_size の半分が入力のサイズになる（fft をかけているため）
    if data_type == "spectrum":
        shape = (int(kernel_size / 2), 1)
    elif data_type == "spectrogram":
        shape = (128, 512, 1)
    else:
        print("correct here based on your model")
        sys.exit(1)

    inputs = tf.keras.Input(shape=shape)
    # hidden_shape is (batch, 192)
    hidden = spectrum_conv(
        x=inputs,
        has_attention=has_attention,
        has_inception=has_inception,
        is_mul_layer=is_mul_layer,
    )
    hidden_inputs = tf.keras.Input(shape=(192,))
    # NOTE: main の方は 特徴量空間の変換を持たない
    output_main = classifier4enn(
        x=hidden_inputs,
        has_dropout=has_dropout,
        hidden_dim=0,
        has_converted_space=False,
        n_class=n_class,
    )
    output_sub = classifier4enn(
        x=hidden_inputs,
        has_dropout=has_dropout,
        hidden_dim=192,
        has_converted_space=True,
        n_class=n_class,
    )

    exploit_model = tf.keras.Model(inputs=inputs, outputs=hidden)
    classifier_main_model = tf.keras.Model(
        inputs=hidden_inputs, outputs=output_main
    )
    classifier_sub_model = tf.keras.Model(
        inputs=hidden_inputs, outputs=output_sub
    )
    # 最適化関数の設定
    optimizer = tf.keras.optimizers.Adam()
    # メトリクスの作成
    # true side : カテゴリカルな状態，pred side : one-hot表現（ソフトマックスをかける前）
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    epoch_loss_main_avg = tf.keras.metrics.Mean()
    epoch_loss_sub_avg = tf.keras.metrics.Mean()
    loss_class = EDLLoss(K=n_class, annealing=0)
    # サマリーライターのセットアップ
    # current_time = date_id
    # train_log_dir = os.path.join(
    #     os.environ["sleep"], "logs", "gradient_tape", current_time, "train"
    # )
    # train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # エポックのループ
    for epoch in range(epochs):
        # ロス関数はエポックごとにアニーリングを変えるので中に書く
        loss_class.annealing = min(1, annealing_param * (epoch / epochs))
        print(f"エポック:{epoch + 1}")
        # エポック内のバッチサイズごとのループ
        for _, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            # 勾配を計算
            with tf.GradientTape(persistent=True) as tape_main:
                hidden_main = exploit_model(x_batch_train, training=True)
                evidence_main = classifier_main_model(
                    hidden_main, training=True
                )
                alpha_main = evidence_main + 1
                y_pred_train = alpha_main / tf.reduce_sum(
                    alpha_main, axis=1, keepdims=True
                )
                unc_main = n_class / tf.reduce_sum(
                    alpha_main, axis=1, keepdims=True
                )
                loss_value_main = loss_class.call(
                    tf.keras.utils.to_categorical(
                        y_batch_train, num_classes=n_class
                    ),
                    evidence_main,
                )
                if epoch / epochs > subnet_starting_point:
                    evidence_sub = classifier_sub_model(
                        hidden_main, training=True
                    )
                    alpha_sub = evidence_sub + 1
                    y_pred_sub = alpha_sub / tf.reduce_sum(
                        alpha_sub, axis=1, keepdims=True
                    )
                    loss_value_sub = loss_class.call(
                        tf.keras.utils.to_categorical(
                            y_batch_train, num_classes=n_class
                        ),
                        evidence_sub,
                        unc_main,
                    )
                    # 進捗の記録
                    epoch_loss_sub_avg(loss_value_sub)
            if epoch / epochs > subnet_starting_point:
                grads_sub = tape_main.gradient(
                    loss_value_sub, classifier_sub_model.trainable_weights
                )
                optimizer.apply_gradients(
                    zip(grads_sub, classifier_sub_model.trainable_weights)
                )
                y_merged = (
                    1 - unc_main
                ) * y_pred_train + unc_main * y_pred_sub
                train_acc_metric.update_state(y_batch_train, y_merged)
            else:
                train_acc_metric.update_state(y_batch_train, y_pred_train)

            [grads_main, grads_exploit] = tape_main.gradient(
                loss_value_main,
                [
                    classifier_main_model.trainable_weights,
                    exploit_model.trainable_weights,
                ],
            )
            optimizer.apply_gradients(
                zip(grads_main, classifier_main_model.trainable_weights)
            )
            optimizer.apply_gradients(
                zip(grads_exploit, exploit_model.trainable_weights)
            )
            # 進捗の記録
            epoch_loss_main_avg(loss_value_main)

        print(
            f"訓練一致率：{train_acc_metric.result():.2%}",
            f"訓練損失(main):{epoch_loss_main_avg.result():.5f}",
            f"訓練損失(merged):{epoch_loss_sub_avg.result():.5f}",
        )
        # wandbにログを送る（TODO：pre, rec, f-mも送る)
        # TODO: 一か所にまとめる or commit false にする
        # エポックの終わりに訓練メトリクスを初期化
        train_acc_metric.reset_states()

        # 訓練の終わりに検証用データ（今回はテストデータ）の性能を見る
        for x_batch_val, y_batch_val in val_dataset:
            hidden_main = exploit_model(x_batch_val, training=False)
            evidence_main = classifier_main_model(hidden_main, training=False)
            alpha_main = evidence_main + 1
            y_pred_main = alpha_main / tf.reduce_sum(
                alpha_main, axis=1, keepdims=True
            )
            unc_main = n_class / tf.reduce_sum(
                alpha_main, axis=1, keepdims=True
            )
            if epoch / epochs > subnet_starting_point:
                evidence_sub = classifier_sub_model(
                    hidden_main, training=False
                )
                alpha_sub = evidence_sub + 1
                y_pred_sub = alpha_sub / tf.reduce_sum(
                    alpha_main, axis=1, keepdims=True
                )
                y_merged = (1 - unc_main) * y_pred_main + unc_main * y_pred_sub
                val_acc_metric.update_state(y_batch_val, y_merged)
            else:
                val_acc_metric.update_state(y_batch_val, y_pred_main)
        # メトリクスを表示
        val_acc = val_acc_metric.result()
        print(f"テスト一致率：{val_acc:.2%}")
        # 初期化
        val_acc_metric.reset_states()

        # TODO: 各睡眠段階の一致率・再現率・適合率・F値を計算する
        # (each_ss_acc, rec, pre, f_m) = utils.calc_ss_prop()

        # wandbにログを送る（TODO：pre, rec, f-mも送る)
        log_info = {
            "train_acc": train_acc_metric.result(),
            "train_loss_main": epoch_loss_main_avg.result(),
            "train_loss_sub": epoch_loss_sub_avg.result(),
            "val_acc": val_acc_metric.result(),
        }
        wandb.log(log_info)

    # モデルの保存
    path = os.path.join(pre_process.my_env.models_dir, test_name)
    exploit_model_path = os.path.join(path, "exploit")
    main_model_path = os.path.join(path, "main")
    sub_model_path = os.path.join(path, "sub")
    if not os.path.exists(exploit_model_path):
        os.makedirs(exploit_model_path)
    if not os.path.exists(main_model_path):
        os.makedirs(main_model_path)
    if not os.path.exists(sub_model_path):
        os.makedirs(sub_model_path)

    exploit_model.save(exploit_model_path + f"/{date_id}")
    classifier_main_model.save(main_model_path + f"/{date_id}")
    classifier_sub_model.save(sub_model_path + f"/{date_id}")
    # # wandbへの記録終了
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

    # ANCHOR
    # ハイパーパラメータの設定
    TEST_RUN = True
    HAS_ATTENTION = True
    PSE_DATA = False
    HAS_INCEPTION = True
    IS_PREVIOUS = False
    IS_NORMAL = True
    IS_ENN = False
    EPOCHS = 100
    BATCH_SIZE = 512
    N_CLASS = 5
    KERNEL_SIZE = 512
    STRIDE = 1024
    SAMPLE_SIZE = 5000
    ANNEALING_RATIO = 16
    SUBNET_STARTING_POINNT = 0.5
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
            name=test_name,
            project=WANDB_PROJECT,
            pre_process=pre_process,
            train=train,
            test=test,
            epochs=EPOCHS,
            has_attention=HAS_ATTENTION,
            my_tags=my_tags,
            date_id=date_id,
            pse_data=PSE_DATA,
            test_name=test_name,
            kernel_size=KERNEL_SIZE,
            wandb_config=wandb_config,
            sample_size=SAMPLE_SIZE,
            is_mul_layer=False,
            has_dropout=False,
            subnet_starting_point=SUBNET_STARTING_POINNT,
            annealing_param=ANNEALING_RATIO,
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
