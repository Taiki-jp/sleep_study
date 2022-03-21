import datetime
import os
import sys
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.metrics import (
    BinaryAccuracy,  # TrueNegatives,  # TN; TruePositives,  # TP; FalseNegatives,  # FN; FalsePositives,  # FP; 一致率
)
from tensorflow.keras.metrics import Precision  # Precision
from tensorflow.keras.metrics import Recall  # Recall
from tensorflow.python.ops.numpy_ops.np_math_ops import positive

from data_analysis.py_color import PyColor
from data_analysis.utils import Utils
from nn.losses import EDLLoss
from nn.model_base import EDLModelBase, edl_classifier_1d, edl_classifier_2d
from nn.utils import load_bin_model, separate_unc_data
from pre_process.json_base import JsonBase
from pre_process.main_param_reader import MainParamReader
from pre_process.pre_process import PreProcess
from pre_process.utils import set_seed


def main(
    is_under_4hz: bool,
    date_id_for_save: str,
    pre_process: PreProcess,
    pse_data: bool,
    target_ss: str,
    test: list,
    train: list,
    utils: Utils,
    batch_size: int = 0,
    date_id: dict = dict(),
    n_class: int = 5,
    sample_size: int = 0,
    test_name: str = None,
    unc_threthold: float = 0,
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
        is_shuffle=False,
        pse_data=pse_data,
        to_one_hot_vector=False,
        each_data_size=sample_size,
        class_size=5,  # ここは予測するラベル数ではなく，睡眠段階の数を指定している
        n_class_converted=n_class,
        target_ss=[target_ss],
        is_under_4hz=is_under_4hz,
    )

    # データクレンジングを行うベースとなるモデルを読み込む
    model = load_bin_model(
        loaded_name=test_name,
        verbose=0,
        is_all=False,
        ss_id=date_id,
        ss=target_ss,
    )
    # モデルが一つでもない場合はreturn
    if any(model) is None:
        print(PyColor.RED_FLASH, "modelが空です", PyColor.END)
        sys.exit(1)

    shape: Tuple[int, int, int] = (64, 30, 1)
    inputs = tf.keras.Input(shape=shape)
    outputs = edl_classifier_2d(
        x=inputs,
        n_class=n_class,
        has_attention=False,
        has_inception=True,
        is_mul_layer=False,
        has_dropout=True,
        dropout_rate=0.3,
    )
    # catching_flag_train = utils.stop_early(y=_y, mode="catching_assertion")
    # catching_flag_test = utils.stop_early(y=_y_test, mode="catching_assertion")
    # 早期終了フラグが立った場合はそこで終了
    # if catching_flag_train or catching_flag_test:
    #     return
    # 訓練データのクレンジング
    (_x, _y) = separate_unc_data(
        x=x_train,
        y=y_train[0],
        model=model[0],
        batch_size=batch_size,
        n_class=n_class,
        experiment_type="positive_cleansing",
        unc_threthold=unc_threthold,
        verbose=0,
    )
    # 検証データのクレンジング
    (_x_val, _y_val) = separate_unc_data(
        x=x_val,
        y=y_val[0],
        model=model[0],
        batch_size=batch_size,
        n_class=n_class,
        experiment_type="positive_cleansing",
        unc_threthold=unc_threthold,
        verbose=0,
    )
    # テストデータのクレンジング
    (_x_test, _y_test) = separate_unc_data(
        x=x_test,
        y=y_test[0],
        model=model[0],
        batch_size=batch_size,
        n_class=n_class,
        experiment_type="positive_cleansing",
        unc_threthold=unc_threthold,
        verbose=0,
    )
    # ennの時はevidenceから計算する
    sub_model = EDLModelBase(inputs=inputs, outputs=outputs, n_class=n_class)
    sub_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=EDLLoss(K=n_class, annealing=0.1),
        metrics=[
            BinaryAccuracy(name="bn"),
            Precision(name="precision"),
            Recall(name="recall"),
        ],
    )
    cp_dir = pre_process.my_env.get_model_saved_path(
        c_dir=test_name,
        ss_dir=target_ss,
        model_id=date_id_for_save,
    )
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=cp_dir,
        verbose=1,
        period=1,
        monitor="val_bn",
        save_best_only=True,
        mode="max",
    )
    sub_model.fit(
        x=_x,
        y=_y,
        batch_size=batch_size,
        validation_data=(_x_val, _y_val),
        epochs=10,
        callbacks=[cp_callback],
        verbose=2,
    )

    # ENN（サブ）の場合正解データ、予測データ、不確実性を書き出す
    evidence = sub_model.predict(x_test)
    can_save = utils.output_enn_pred(
        evidence=evidence,
        y_true=y_test[0],
        target_ss=target_ss,
        test_name=test_name,
        is_double=True,
    )
    if can_save:
        return
    else:
        print(
            PyColor.RED_FLASH, "cannot save model prediction file", PyColor.END
        )


if __name__ == "__main__":
    # シードの固定
    set_seed(0)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # 環境設定
    CALC_DEVICE = "gpu"
    DEVICE_ID = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID
    tf.keras.backend.set_floatx("float32")
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # 実験用パラメータを読み込む
    MPR = MainParamReader()

    # ANCHOR: ハイパーパラメータの設定
    BATCH_SIZE = MPR.denn_ensemble["batch_size"]
    CLEANSING_TYPE = MPR.denn_ensemble["cleansing_type"]
    DATA_TYPE = MPR.denn_ensemble["data_type"]
    DROPOUT_RATE = MPR.denn_ensemble["dropout_rate"]
    EPOCHS = MPR.denn_ensemble["epochs"]
    FIT_POS = MPR.denn_ensemble["fit_pos"]
    HAS_ATTENTION = MPR.denn_ensemble["has_attention"]
    HAS_DROPOUT = MPR.denn_ensemble["has_dropout"]
    HAS_INCEPTION = MPR.denn_ensemble["has_inception"]
    HAS_NREM2_BIAS = MPR.denn_ensemble["has_nrem2_bias"]
    HAS_REM_BIAS = MPR.denn_ensemble["has_rem_bias"]
    INCEPTION_TAG = "inception" if HAS_INCEPTION else "no-inception"
    IS_ENN = MPR.denn_ensemble["is_enn"]
    IS_MUL_LAYER = MPR.denn_ensemble["is_mul_layer"]
    IS_NORMAL = MPR.denn_ensemble["is_normal"]
    IS_PREVIOUS = MPR.denn_ensemble["is_previous"]
    IS_SIMPLE_RNN = MPR.denn_ensemble["is_simple_rnn"]
    IS_TIME_SERIES = MPR.denn_ensemble["is_time_series"]
    IS_UNDER_4HZ = MPR.denn_ensemble["is_under_4hz"]
    KERNEL_SIZE = MPR.denn_ensemble["kernel_size"]
    NORMAL_TAG = "normal" if IS_NORMAL else "sas"
    N_CLASS = MPR.denn_ensemble["n_class"]
    PSE_DATA = MPR.denn_ensemble["pse_data"]
    PSE_DATA_TAG = "psedata" if PSE_DATA else "sleepdata"
    SAMPLE_SIZE = MPR.denn_ensemble["sample_size"]
    SAVE_MODEL = MPR.denn_ensemble["save_model"]
    STRIDE = MPR.denn_ensemble["stride"]
    TEST_RUN = MPR.denn_ensemble["test_run"]
    TARGET_SS = MPR.denn_ensemble[
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
    # 読み込むモデルの日付リストを返す
    MI = pre_process.my_env.mi
    datasets = pre_process.load_sleep_data.load_data(
        load_all=True, pse_data=False
    )

    # ここでは69名全員が呼ばれるが，後のpreprocess.namelistで被験者が絞られるので大丈夫
    model_date_list = MI.make_model_id_list4bin_format()

    # モデルのidを記録するためのリスト
    date_id_saving_list: List[str] = list()

    for (test_id, test_name) in enumerate(pre_process.name_list):
        # モデルのidを記録するためのリスト
        for target_ss in TARGET_SS:
            date_id_for_save = datetime.datetime.now().strftime(
                "%Y%m%d-%H%M%S"
            )
            (train, test) = pre_process.split_train_test_from_records(
                datasets, test_id=test_id, pse_data=PSE_DATA
            )
            date_id = model_date_list[test_name][target_ss]
            main(
                batch_size=BATCH_SIZE,
                date_id=date_id,
                date_id_for_save=date_id_for_save,
                is_under_4hz=IS_UNDER_4HZ,
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
            MI.dump(
                value=date_id_for_save,
                test_name=test_name,
                target_ss=target_ss,
                cleansing_type="positive_cleansing",
            )

        if TEST_RUN:
            print(PyColor.RED_FLASH, "テストランのため被験者ループを終了します", PyColor.END)
            break
