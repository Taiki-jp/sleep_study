import datetime
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

import tensorflow as tf
from tensorflow.keras.metrics import Precision  # Precision
from tensorflow.keras.metrics import Recall  # Recall
from tensorflow.keras.metrics import BinaryAccuracy

from data_analysis.py_color import PyColor
from data_analysis.utils import Utils
from nn.losses import EDLLoss
from nn.model_base import EDLModelBase, edl_classifier_1d, edl_classifier_2d
from pre_process.main_param_reader import MainParamReader
from pre_process.pre_process import PreProcess, Record
from pre_process.utils import set_seed


def main(
    batch_size: int,
    data_type: str,
    date_id: str,
    dropout_rate: float,
    epochs: int,
    has_attention: bool,
    has_dropout: bool,
    has_inception: bool,
    is_enn: bool,
    is_mul_layer: bool,
    is_under_4hz: bool,
    kernel_size: int,
    n_class: int,
    pre_process: PreProcess,
    pse_data: bool,
    sample_size: int,
    target_ss: str,
    test: List[Record],
    test_name: str,
    train: List[Record],
    utils: Utils,
):

    # データセットの作成
    (
        (x_train, y_train),
        (x_val, y_val),
        (x_test, y_test),
    ) = pre_process.make_dataset(
        class_size=5,  # NOTE: ここは予測するラベル数ではなく，睡眠段階の数を指定している
        each_data_size=sample_size,
        is_shuffle=True,
        is_storchastic=False,
        is_under_4hz=is_under_4hz,
        n_class_converted=n_class,
        pse_data=pse_data,
        target_ss=[target_ss],  # FIXME: ここをリストで指定する必要ある？
        test=test,
        to_one_hot_vector=False,
        train=train,
    )

    # モデルの作成とコンパイル
    # NOTE: kernel_size の半分が入力のサイズになる（fft をかけているため）
    # TODO: メソッド化
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
        model = EDLModelBase(inputs=inputs, outputs=outputs, n_class=n_class)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=EDLLoss(K=n_class, annealing=0.1),
            metrics=[
                BinaryAccuracy(name="bn"),
                Precision(name="precision"),
                Recall(name="recall"),
            ],
        )
        monitoring = "val_bn"

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
        monitoring = "val_accuracy"

    # tensorboard, model保存のコールバック作成
    log_dir = pre_process.my_env.get_tf_board_saved_path(
        p_dir="logs", c_dir=test_name, model_id=date_id
    )
    tf_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )
    cp_dir = pre_process.my_env.get_model_saved_path(
        c_dir=test_name,
        ss_dir=target_ss,
        model_id=date_id,
    )
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=cp_dir,
        verbose=1,
        period=1,
        monitor=monitoring,
        save_best_only=True,
        mode="max",
    )

    model.fit(
        x_train,
        y_train[0],
        batch_size=batch_size,
        validation_data=(x_val, y_val[0]),
        epochs=epochs,
        callbacks=[
            tf_callback,
            cp_callback,
        ],
        verbose=2,
    )

    # ENNの場合正解データ，予測データ，不確実性を書き出す
    if is_enn:
        evidence = model.predict(x_test)
        can_save = utils.output_enn_pred(
            evidence=evidence,
            y_true=y_test[0],
            target_ss=target_ss,
            test_name=test_name,
            is_double=False,
        )
        if can_save:
            return
        else:
            print(
                PyColor.RED_FLASH,
                "cannot save model prediction file",
                PyColor.END,
            )


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
    # 実験用パラメータを読み込む
    MPR = MainParamReader()

    BATCH_SIZE = MPR.main_setting["batch_size"]
    CLEANSING_TYPE = MPR.main_setting["cleansing_type"]
    DATA_TYPE = MPR.main_setting["data_type"]
    DROPOUT_RATE = MPR.main_setting["dropout_rate"]
    EPOCHS = MPR.main_setting["epochs"]
    FIT_POS = MPR.main_setting["fit_pos"]
    HAS_ATTENTION = MPR.main_setting["has_attention"]
    HAS_DROPOUT = MPR.main_setting["has_dropout"]
    HAS_INCEPTION = MPR.main_setting["has_inception"]
    HAS_NREM2_BIAS = MPR.main_setting["has_nrem2_bias"]
    HAS_REM_BIAS = MPR.main_setting["has_rem_bias"]
    INCEPTION_TAG = "inception" if HAS_INCEPTION else "no-inception"
    IS_ENN = MPR.main_setting["is_enn"]
    IS_MUL_LAYER = MPR.main_setting["is_mul_layer"]
    IS_NORMAL = MPR.main_setting["is_normal"]
    IS_PREVIOUS = MPR.main_setting["is_previous"]
    IS_SIMPLE_RNN = MPR.main_setting["is_simple_rnn"]
    IS_TIME_SERIES = MPR.main_setting["is_time_series"]
    IS_UNDER_4HZ = MPR.main_setting["is_under_4hz"]
    KERNEL_SIZE = MPR.main_setting["kernel_size"]
    NORMAL_TAG = "normal" if IS_NORMAL else "sas"
    N_CLASS = MPR.main_setting["n_class"]
    PSE_DATA = MPR.main_setting["pse_data"]
    PSE_DATA_TAG = "psedata" if PSE_DATA else "sleepdata"
    SAMPLE_SIZE = MPR.main_setting["sample_size"]
    SAVE_MODEL = MPR.main_setting["save_model"]
    STRIDE = MPR.main_setting["stride"]
    TEST_RUN = MPR.main_setting["test_run"]
    TARGET_SS = MPR.main_setting[
        "target_ss"
    ]  # target_ss としてpre_process.change_labelでnr3として扱いたいのでnr4, nr34とはしていない
    ATTENTION_TAG = "attention" if HAS_ATTENTION else "no-attention"
    ENN_TAG = "enn" if IS_ENN else "dnn"
    INCEPTION_TAG += "v2" if IS_MUL_LAYER else ""

    # オブジェクトの作成
    pre_process = PreProcess(
        cleansing_type=CLEANSING_TYPE,
        data_type=DATA_TYPE,
        fit_pos=FIT_POS,
        has_ignored=True,
        has_nrem2_bias=HAS_NREM2_BIAS,
        has_rem_bias=HAS_REM_BIAS,
        is_normal=IS_NORMAL,
        is_previous=IS_PREVIOUS,
        kernel_size=KERNEL_SIZE,
        lsp_option="nr2",
        make_valdata=True,
        model_type=ENN_TAG,
        stride=STRIDE,
        verbose=0,
    )
    # 記録用のjsonファイルを読み込む
    MI = pre_process.my_env.mi
    datasets = pre_process.load_sleep_data.load_data(
        load_all=True,
        pse_data=PSE_DATA,
    )

    start = 0
    for test_id, test_name in zip(
        range(start, len(pre_process.name_list)), pre_process.name_list[start:]
    ):
        # モデルのidを記録するためのリスト
        for target_ss in TARGET_SS:
            date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            (train, test) = pre_process.split_train_test_from_records(
                datasets, test_id=test_id, pse_data=PSE_DATA
            )
            main(
                batch_size=BATCH_SIZE,
                data_type=DATA_TYPE,
                date_id=date_id,
                dropout_rate=DROPOUT_RATE,
                epochs=EPOCHS,
                has_attention=HAS_ATTENTION,
                has_dropout=True,
                has_inception=HAS_INCEPTION,
                is_enn=IS_ENN,
                is_mul_layer=IS_MUL_LAYER,
                is_under_4hz=IS_UNDER_4HZ,
                kernel_size=KERNEL_SIZE,
                n_class=N_CLASS,
                pre_process=pre_process,
                pse_data=PSE_DATA,
                sample_size=SAMPLE_SIZE,
                target_ss=target_ss,
                test=test,
                test_name=test_name,
                train=train,
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
            )

            # json に書き込み
            MI.dump(value=date_id, test_name=test_name, target_ss=target_ss)

        if TEST_RUN:
            print(PyColor.RED_FLASH, "テストランのため被験者ループを終了します", PyColor.END)
            break
