import datetime
import os
import sys
from collections import Counter

import tensorflow as tf
from tensorflow.keras.metrics import (
    BinaryAccuracy,
    FalseNegatives,
    FalsePositives,
    Precision,
    Recall,
    TrueNegatives,
    TruePositives,
)

import wandb
from data_analysis.py_color import PyColor
from data_analysis.utils import Utils
from nn.losses import EDLLoss
from nn.model_base import EDLModelBase, edl_classifier_1d, edl_classifier_2d
from nn.utils import set_seed
from nn.wandb_classification_callback import WandbClassificationCallback
from pre_process.json_base import JsonBase
from pre_process.pre_process import PreProcess


def main(
    is_simple_rnn: bool,
    test_run: bool,
    dropout_rate: float,
    has_dropout: bool,
    log_tf_projector: bool,
    name: str,
    project: str,
    train: list,
    test: list,
    pre_process: PreProcess,
    epochs: int = 1,
    save_model: bool = False,
    my_tags: list = None,
    batch_size: int = 32,
    n_class: int = 5,
    pse_data: bool = False,
    test_name: str = None,
    date_id: str = None,
    has_attention: bool = False,
    has_inception: bool = True,
    data_type: str = None,
    sample_size: int = 0,
    is_enn: bool = True,
    wandb_config: dict = dict(),
    kernel_size: int = 0,
    is_mul_layer: bool = False,
    utils: Utils = None,
    target_ss: str = "",
):

    # データセットの作成
    (x_train, y_train), (x_test, y_test) = pre_process.make_dataset(
        train=train,
        test=test,
        is_storchastic=False,
        is_shuffle=False,
        pse_data=pse_data,
        to_one_hot_vector=False,
        each_data_size=sample_size,
        class_size=5,  # ここは予測するラベル数ではなく，睡眠段階の数を指定している
        n_class_converted=n_class,
        target_ss=[target_ss],
    )
    # データセットの数を表示
    print(f"training data : {x_train.shape}")
    ss_train_dict = Counter(y_train)
    ss_test_dict = Counter(y_test)
    print(PyColor.GREEN_FLASH, "train:", ss_train_dict, PyColor.END)
    print(PyColor.GREEN_FLASH, "test:", ss_test_dict, PyColor.END)

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
    if data_type == "spectrum" or data_type == "cepstrum":
        shape = (int(kernel_size / 2), 1)
    elif data_type == "spectrogram":
        shape = (128, 30, 1)
    else:
        print("correct here based on your model")
        sys.exit(1)

    inputs = tf.keras.Input(shape=shape)
    if data_type == "spectrum" or data_type == "cepstrum":
        outputs = edl_classifier_1d(
            x=inputs,
            n_class=n_class,
            has_attention=has_attention,
            has_inception=has_inception,
            is_mul_layer=is_mul_layer,
            has_dropout=has_dropout,
            dropout_rate=dropout_rate,
            is_simple_rnn=is_simple_rnn,
        )
    elif data_type == "spectrogram":
        outputs = edl_classifier_2d(
            x=inputs,
            n_class=n_class,
            has_attention=has_attention,
            has_inception=has_inception,
            is_mul_layer=is_mul_layer,
            has_dropout=has_dropout,
            dropout_rate=dropout_rate,
            is_simple_rnn=is_simple_rnn,
        )
    if is_enn:
        model = EDLModelBase(inputs=inputs, outputs=outputs, n_class=n_class)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=EDLLoss(K=n_class, annealing=0.1),
            metrics=[
                BinaryAccuracy(name="bn"),
                Precision(name="precision"),
                Recall(name="recall"),
                TruePositives(name="tp"),
                FalseNegatives(name="fn"),
                TrueNegatives(name="tn"),
                FalsePositives(name="fp"),
                "mse",
            ],
        )

    else:
        model = tf.keras.Model(inputs=inputs, outputs=outputs, n_class=n_class)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=[
                "accuracy",
                # "mse"
            ],
        )

    # tensorboard作成
    log_dir = os.path.join(
        pre_process.my_env.project_dir, "my_edl", test_name, date_id
    )
    # tf_callback = tf.keras.callbacks.TensorBoard(
    #     log_dir=log_dir, histogram_freq=1
    # )

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        epochs=epochs,
        callbacks=[
            # tf_callback,
            WandbClassificationCallback(
                validation_data=(x_test, y_test),
                log_confusion_matrix=True,
                labels=["nr34", "nr2", "nr1", "rem", "wake"]
                if n_class == 5
                else ["non_target", "target"],
            ),
        ],
        verbose=2,
    )
    # 混合行列・不確かさ・ヒストグラムの作成
    tuple_x = (x_train, x_test)
    tuple_y = (y_train, y_test)
    for train_or_test, _x, _y in zip(["train", "test"], tuple_x, tuple_y):
        evidence = model.predict(_x)
        utils.make_graphs(
            y=_y,
            evidence=evidence,
            train_or_test=train_or_test,
            graph_person_id=test_name,
            calling_graph="all",
            graph_date_id=date_id,
            is_each_unc=False,
            n_class=n_class,
            norm_cm=True,
            is_joinplot=False,
        )
    # tensorboardのログ
    # if log_tf_projector:
    #     utils.make_tf_projector(
    #         x=x_test,
    #         y=y_test,
    #         batch_size=batch_size,
    #         hidden_layer_id=-7,
    #         log_dir=log_dir,
    #         data_type=data_type,
    #         model=model,
    #     )

    if save_model is True and test_run is False:
        print(PyColor().GREEN_FLASH, "モデルを保存します ...", PyColor().END)
        path = os.path.join(pre_process.my_env.models_dir, test_name, date_id)
        model.save(path)
    wandb.finish()


if __name__ == "__main__":
    # 環境設定
    CALC_DEVICE = "gpu"
    DEVICE_ID = "0" if CALC_DEVICE == "gpu" else "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID
    if os.environ["CUDA_VISIBLE_DEVICES"] != "-1":
        tf.keras.backend.set_floatx("float32")
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # tf.config.run_functions_eagerly(True)
    else:
        print("*** cpuで計算します ***")
        tf.config.run_functions_eagerly(True)

    # ハイパーパラメータの設定
    TEST_RUN = False
    EPOCHS = 0
    HAS_ATTENTION = True
    PSE_DATA = False
    HAS_INCEPTION = True
    IS_PREVIOUS = False
    IS_NORMAL = True
    HAS_DROPOUT = True
    IS_ENN = True
    # FIXME: 多層化はとりあえずいらない
    IS_MUL_LAYER = True
    HAS_NREM2_BIAS = False
    HAS_REM_BIAS = False
    SAVE_MODEL = False
    IS_TIME_SERIES = False
    IS_SIMPLE_RNN = False
    DROPOUT_RATE = 0.2
    BATCH_SIZE = 64
    N_CLASS = 2
    SAMPLE_SIZE = 1000
    # TARGET_SS = ["nr2"]
    TARGET_SS = ["wake", "rem", "nr1", "nr2", "nr3"]
    DATA_TYPE = "spectrogram"
    STRIDE = 16 if DATA_TYPE == "spectrogram" else 480
    KERNEL_SIZE = 256 if DATA_TYPE == "spectrogram" else 512
    FIT_POS = "middle"
    NORMAL_TAG = "normal" if IS_NORMAL else "sas"
    ATTENTION_TAG = "attention" if HAS_ATTENTION else "no-attention"
    PSE_DATA_TAG = "psedata" if PSE_DATA else "sleepdata"
    INCEPTION_TAG = "inception" if HAS_INCEPTION else "no-inception"
    # WANDB_PROJECT = "test" if TEST_RUN else "master"
    WANDB_PROJECT = "test" if TEST_RUN else "bin_classification"
    ENN_TAG = "enn" if IS_ENN else "dnn"
    INCEPTION_TAG += "v2" if IS_MUL_LAYER else ""

    # シードの固定
    set_seed(0)

    # 記録用のjsonファイルを読み込む
    JB = JsonBase("model_id.json")
    JB.load()
    # オブジェクトの作成
    pre_process = PreProcess(
        data_type=DATA_TYPE,
        fit_pos=FIT_POS,
        verbose=0,
        kernel_size=KERNEL_SIZE,
        is_previous=IS_PREVIOUS,
        stride=STRIDE,
        is_normal=IS_NORMAL,
        has_nrem2_bias=HAS_NREM2_BIAS,
        has_rem_bias=HAS_REM_BIAS,
        is_time_series=IS_TIME_SERIES,
    )
    datasets = pre_process.load_sleep_data.load_data(
        load_all=True,
        pse_data=PSE_DATA,
    )
    # モデルのidを記録するためのリスト
    date_id_saving_list = list()

    for target_ss in TARGET_SS:
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
                f"kernel:{KERNEL_SIZE}",
                f"stride:{STRIDE}",
                f"sample:{SAMPLE_SIZE}",
                f"model:{ENN_TAG}",
                f"epoch:{EPOCHS}",
                f"nrem2_bias:{HAS_NREM2_BIAS}",
                f"rem_bias:{HAS_REM_BIAS}",
                f"dropout:{HAS_DROPOUT}:rate{DROPOUT_RATE}",
                target_ss,
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
                "has_nrem2_bias": HAS_NREM2_BIAS,
                "has_rem_bias": HAS_REM_BIAS,
                "model_type": ENN_TAG,
                "data_type": DATA_TYPE,
                "target_ss": target_ss,
            }
            main(
                test_run=TEST_RUN,
                has_dropout=True,
                log_tf_projector=False,
                name=test_name,
                project=WANDB_PROJECT,
                pre_process=pre_process,
                train=train,
                test=test,
                epochs=EPOCHS,
                save_model=SAVE_MODEL,
                has_attention=HAS_ATTENTION,
                my_tags=my_tags,
                date_id=date_id,
                pse_data=PSE_DATA,
                test_name=test_name,
                has_inception=HAS_INCEPTION,
                batch_size=BATCH_SIZE,
                n_class=N_CLASS,
                data_type=DATA_TYPE,
                sample_size=SAMPLE_SIZE,
                is_enn=IS_ENN,
                wandb_config=wandb_config,
                kernel_size=KERNEL_SIZE,
                is_mul_layer=IS_MUL_LAYER,
                utils=Utils(),
                dropout_rate=DROPOUT_RATE,
                target_ss=target_ss,
                is_simple_rnn=IS_SIMPLE_RNN,
            )
            # testの時は一人の被験者で止める
            if TEST_RUN:
                print(PyColor.RED_FLASH, "テストランのため終了します01", PyColor.END)
                break
        if TEST_RUN:
            print(PyColor.RED_FLASH, "テストランのため終了します02", PyColor.END)
            break

        if SAVE_MODEL == True and TEST_RUN == False:
            # json に書き込み
            JB.dump(
                keys=[
                    JB.first_key_of_pre_process(
                        is_normal=IS_NORMAL, is_prev=IS_PREVIOUS
                    ),
                    ENN_TAG,
                    DATA_TYPE,
                    FIT_POS,
                    f"stride_{str(STRIDE)}",
                    f"kernel_{str(KERNEL_SIZE)}",
                    "no_cleansing",
                    f"{target_ss}",
                ],
                value=date_id_saving_list,
            )
