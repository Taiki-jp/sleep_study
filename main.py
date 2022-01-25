import datetime
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

import tensorflow as tf

import wandb
from data_analysis.py_color import PyColor
from data_analysis.utils import Utils
from nn.losses import EDLLoss
from nn.model_base import EDLModelBase, edl_classifier_1d, edl_classifier_2d
from nn.wandb_classification_callback import WandbClassificationCallback

# from nn.metrics import CategoricalTruePositives
from pre_process.pre_process import PreProcess, Record
from pre_process.utils import set_seed

# from wandb.keras import WandbCallback


def main(
    dropout_rate: float,
    has_dropout: bool,
    log_tf_projector: bool,
    name: str,
    project: str,
    train: List[Record],
    test: List[Record],
    pre_process: PreProcess,
    epochs: int = 1,
    save_model: bool = False,
    my_tags: List[str] = None,
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
    wandb_config: Dict[str, Any] = dict(),
    kernel_size: int = 0,
    is_mul_layer: bool = False,
    utils: Utils = None,
    is_under_4hz: bool = False,
):

    # データセットの作成
    (
        (x_train, y_train),
        (x_val, y_val),
        (x_test, y_test),
    ) = pre_process.make_dataset(
        train=train,
        test=test,
        is_storchastic=False,
        pse_data=pse_data,
        to_one_hot_vector=False,
        each_data_size=sample_size,
        is_under_4hz=is_under_4hz,
    )
    # データセットの数を表示
    print(f"training data : {x_train.shape}")
    ss_train_dict: Dict[int, int] = Counter(y_train[0, :])
    ss_test_dict: Dict[int, int] = Counter(y_test[0, :])

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
        shape: Tuple[int, int] = (int(kernel_size / 2), 1)
    elif data_type == "spectrogram":
        if is_under_4hz:
            shape: Tuple[int, int, int] = (
                (32, 30, 1) if kernel_size == 128 else (64, 30, 1)
            )
        else:
            shape: Tuple[int, int, int] = (
                (64, 30, 1) if kernel_size == 128 else (128, 30, 1)
            )
    else:
        print("correct data_type based on your model")
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
        )
    if is_enn:
        model = EDLModelBase(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=EDLLoss(K=n_class, annealing=0.1),
            metrics=["accuracy"],
        )

    else:
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=[
                "accuracy",
            ],
        )

    # tensorboard, model保存のコールバック作成
    log_dir = pre_process.my_env.get_tf_board_saved_path(
        p_dir="logs", c_dir=test_name, model_id=date_id
    )
    tf_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )
    cp_dir = pre_process.my_env.get_model_saved_path(
        c_dir=test_name, model_id=date_id
    )
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=cp_dir,
        verbose=1,
        period=1,
        monitor="val_accuracy",
        save_best_only=True,
    )

    model.fit(
        x_train,
        y_train[0],
        batch_size=batch_size,
        validation_data=(x_val, y_val[0]),
        # validation_data=(x_test, y_test[0]),
        epochs=epochs,
        callbacks=[
            tf_callback,
            WandbClassificationCallback(
                validation_data=(x_val, y_val[0]),
                # validation_data=(x_test, y_test[0]),
                log_confusion_matrix=True,
                labels=["nr34", "nr2", "nr1", "rem", "wake"],
            ),
            cp_callback,
        ],
        verbose=2,
    )
    # 混合行列・不確かさ・ヒストグラムの作成
    tuple_x = (x_train, x_val)
    tuple_y = (y_train[0], y_val[0])
    for train_or_test, _x, _y in zip(["train", "test"], tuple_x, tuple_y):
        evidence = model.predict(_x)
        utils.make_graphs(
            y=_y,
            evidence=evidence,
            train_or_test=train_or_test,
            graph_person_id=test_name,
            calling_graph="all",
            graph_date_id=date_id,
            is_each_unc=True,
        )
    # # tensorboardのログ
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

    wandb.finish()


if __name__ == "__main__":
    # シードの固定
    set_seed(0)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # 環境設定
    CALC_DEVICE = "gpu"
    # CALC_DEVICE = "cpu"
    # NOTE: set here to specify which gpu you use
    DEVICE_ID = "0" if CALC_DEVICE == "gpu" else "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID
    # if os.environ["CUDA_VISIBLE_DEVICES"] != "-1":
    if CALC_DEVICE == "gpu":
        tf.keras.backend.set_floatx("float32")
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.config.run_functions_eagerly(True)
    else:
        print("*** cpuで計算します ***")
        # tf.config.run_functions_eagerly(True)

    # ハイパーパラメータの設定
    TEST_RUN = False
    EPOCHS = 25
    HAS_ATTENTION = False
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
    DROPOUT_RATE = 0.2
    BATCH_SIZE = 512
    N_CLASS = 5
    # KERNEL_SIZE = 512
    KERNEL_SIZE = 128
    IS_UNDER_4HZ = False
    # KERNEL_SIZE = 128
    STRIDE = 16
    # STRIDE = 16
    SAMPLE_SIZE = 10000
    DATA_TYPE = "spectrogram"
    FIT_POS = "middle"
    CLEANSING_TYPE = "no_cleansing"
    NORMAL_TAG = "normal" if IS_NORMAL else "sas"
    ATTENTION_TAG = "attention" if HAS_ATTENTION else "no-attention"
    PSE_DATA_TAG = "psedata" if PSE_DATA else "sleepdata"
    INCEPTION_TAG = "inception" if HAS_INCEPTION else "no-inception"
    WANDB_PROJECT = "test" if TEST_RUN else "20220121_enn_normal"
    ENN_TAG = "enn" if IS_ENN else "dnn"
    INCEPTION_TAG += "v2" if IS_MUL_LAYER else ""

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
        model_type=ENN_TAG,
        cleansing_type=CLEANSING_TYPE,
        make_valdata=True,
        has_ignored=True,
        lsp_option="nr2",
    )
    # 記録用のjsonファイルを読み込む
    MI = pre_process.my_env.mi
    datasets = pre_process.load_sleep_data.load_data(
        load_all=True,
        pse_data=PSE_DATA,
    )
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
            f"kernel:{KERNEL_SIZE}",
            # f"stride:{STRIDE}",
            # f"sample:{SAMPLE_SIZE}",
            f"model:{ENN_TAG}",
            # f"epoch:{EPOCHS}",
            f"nrem2_bias:{HAS_NREM2_BIAS}",
            f"rem_bias:{HAS_REM_BIAS}",
            # f"dropout:{HAS_DROPOUT}:rate{DROPOUT_RATE}",
            f"under_4hz:{IS_UNDER_4HZ}",
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
            "under_4hz": IS_UNDER_4HZ,
        }
        main(
            has_dropout=True,
            log_tf_projector=True,
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
            has_inception=HAS_INCEPTION,
            batch_size=BATCH_SIZE,
            n_class=N_CLASS,
            data_type=DATA_TYPE,
            sample_size=SAMPLE_SIZE,
            is_enn=IS_ENN,
            # wandb_config=wandb_config,
            kernel_size=KERNEL_SIZE,
            is_mul_layer=IS_MUL_LAYER,
            utils=Utils(
                IS_NORMAL,
                IS_PREVIOUS,
                DATA_TYPE,
                FIT_POS,
                STRIDE,
                KERNEL_SIZE,
                model_type=ENN_TAG,
                cleansing_type=CLEANSING_TYPE,
            ),
            dropout_rate=DROPOUT_RATE,
            is_under_4hz=IS_UNDER_4HZ,
        )

        # testの時は一人の被験者で止める
        if TEST_RUN:
            break
    # json に書き込み
    MI.dump(value=date_id_saving_list)
