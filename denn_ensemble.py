import datetime
import os
import sys
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops.numpy_ops.np_math_ops import positive

from data_analysis.py_color import PyColor
from data_analysis.utils import Utils
from nn.losses import EDLLoss
from nn.model_base import EDLModelBase, edl_classifier_1d, edl_classifier_2d
from nn.utils import load_bin_model, separate_unc_data
from pre_process.json_base import JsonBase
from pre_process.pre_process import PreProcess
from pre_process.utils import set_seed


def main(
    pse_data: bool,
    target_ss: str,
    is_under_4hz: bool,
    train: list,
    test: list,
    pre_process: PreProcess,
    utils: Utils,
    n_class: int = 5,
    test_name: str = None,
    date_id: dict = dict(),
    sample_size: int = 0,
    batch_size: int = 0,
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
        loaded_name=test_name, verbose=0, is_all=True, ss_id=date_id
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
    catching_flag_train = utils.stop_early(y=_y, mode="catching_assertion")
    catching_flag_test = utils.stop_early(y=_y_test, mode="catching_assertion")
    # 早期終了フラグが立った場合はそこで終了
    if catching_flag_train or catching_flag_test:
        return
    # 5つのモデルでevidenceを出力
    evd_list = list()
    alp_list = list()
    unc_list = list()
    y_pred_list = list()
    for _model in model:
        # 訓練データのクレンジング
        (_x, _y) = separate_unc_data(
            x=x_train,
            y=y_train,
            model=_model,
            batch_size=batch_size,
            n_class=n_class,
            experiment_type="positive_cleansing",
            unc_threthold=unc_threthold,
            verbose=0,
        )
        # 検証データのクレンジング
        (_x_val, _y_val) = separate_unc_data(
            x=x_val,
            y=y_val,
            model=_model,
            batch_size=batch_size,
            n_class=n_class,
            experiment_type="positive_cleansing",
            unc_threthold=unc_threthold,
            verbose=0,
        )
        # テストデータのクレンジング
        (_x_test, _y_test) = separate_unc_data(
            x=x_test,
            y=y_test,
            model=_model,
            batch_size=batch_size,
            n_class=n_class,
            experiment_type="positive_cleansing",
            unc_threthold=unc_threthold,
            verbose=0,
        )
        # ennの時はevidenceから計算する
        sub_model = EDLModelBase(inputs=inputs, outputs=outputs)
        sub_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=EDLLoss(K=n_class, annealing=0.1),
            metrics=[
                "accuracy",
            ],
        )
        log_dir = os.path.join(
            pre_process.my_env.project_dir,
            "denn_ensemble",
            test_name,
            date_id["positive"],
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

    # ANCHOR: ハイパーパラメータの設定
    TEST_RUN = True
    IS_MUL_LAYER = False
    CATCH_NREM2 = True
    IS_NORMAL = True
    HAS_NREM2_BIAS = True
    HAS_REM_BIAS = False
    IS_UNDER_4HZ = False
    BATCH_SIZE = 512
    N_CLASS = 5
    STRIDE = 16
    KERNEL_SIZE = 128
    SAMPLE_SIZE = 10000
    UNC_THRETHOLD = 0.3
    EXPERIMENT_TYPES = (
        "no_cleansing",
        "positive_cleansing",
        "negative_cleansing",
    )
    DATA_TYPE = "spectrogram"
    FIT_POS = "middle"
    IS_PREVIOUS = False
    IS_ENN = False
    PSE_DATA = False
    ENN_TAG = "enn" if IS_ENN else "dnn"
    CLEANSING_TYPE = "no_cleansing"
    TARGET_SS = [
        "wake",
        "rem",
        "nr1",
        "nr2",
        "nr3",
    ]  # target_ss としてpre_process.change_labelでnr3として扱いたいのでnr4, nr34とはしていない
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
    datasets = pre_process.load_sleep_data.load_data(
        load_all=True, pse_data=False
    )

    # 読み込むモデルの日付リストを返す
    MI = pre_process.my_env.mi
    # ここでは69名全員が呼ばれるが，後のpreprocess.namelistで被験者が絞られるので大丈夫
    model_date_list = MI.make_model_id_list4bin_format()

    # モデルのidを記録するためのリスト
    date_id_saving_list: List[str] = list()

    for test_id, test_name in enumerate(pre_process.name_list):
        # モデルのidを記録するためのリスト
        for target_ss in TARGET_SS:
            date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            (train, test) = pre_process.split_train_test_from_records(
                datasets, test_id=test_id, pse_data=PSE_DATA
            )
            main(
                pse_data=PSE_DATA,
                target_ss=target_ss,
                is_under_4hz=IS_UNDER_4HZ,
                log_tf_projector=True,
                name=test_name,
                # project=WANDB_PROJECT,
                pre_process=pre_process,
                train=train,
                test=test,
                epochs=10,
                save_model=False,
                has_attention=False,
                date_id=date_id,
                test_name=test_name,
                has_inception=True,
                batch_size=BATCH_SIZE,
                n_class=N_CLASS,
                data_type=DATA_TYPE,
                sample_size=SAMPLE_SIZE,
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
                dropout_rate=0.3,
            )

            # json に書き込み
            MI.dump(value=date_id, test_name=test_name, target_ss=target_ss)

        if TEST_RUN:
            print(PyColor.RED_FLASH, "テストランのため被験者ループを終了します", PyColor.END)
            break
