import datetime
import os
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from data_analysis.py_color import PyColor
from data_analysis.utils import Utils
from nn.utils import load_model
from pre_process.pre_process import PreProcess
from pre_process.utils import set_seed


# TODO: 使っていない引数の削除
def main(
    train: list,
    test: list,
    pre_process: PreProcess,
    n_class: int = 5,
    pse_data: bool = False,
    test_name: str = None,
    date_id: dict = None,
    sample_size: int = 0,
    batch_size: int = 0,
    utils: Utils = None,
    is_under_4hz: bool = False,
):

    # データセットの作成
    ((_, _), (_, _), (x_test, y_test),) = pre_process.make_dataset(
        train=train,
        test=test,
        is_storchastic=False,
        pse_data=pse_data,
        to_one_hot_vector=False,
        each_data_size=sample_size,
        is_under_4hz=is_under_4hz,
    )

    # 5段階モデルの読み込み
    model = load_model(
        loaded_name=test_name, model_id=date_id, n_class=n_class, verbose=1
    )

    # CNNによる推定
    logit = model.predict(x=x_test, batch_size=batch_size)
    y_pred = np.argmax(logit, axis=1)
    # 睡眠段階を出力
    ss_df = pd.DataFrame({"y_true": y_test[0], "y_pred": y_pred})
    output_path = os.path.join(utils.env.tmp_dir, test_name, "ss_5class.csv")
    ss_df.to_csv(output_path)


if __name__ == "__main__":
    set_seed(0)
    # 環境設定
    CALC_DEVICE = "gpu"
    # CALC_DEVICE = "cpu"
    DEVICE_ID = "0" if CALC_DEVICE == "gpu" else "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DEEETERMINISTIC"] = "1"
    if os.environ["CUDA_VISIBLE_DEVICES"] != "-1":
        tf.keras.backend.set_floatx("float32")
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # tf.config.run_functions_eagerly(True)
    else:
        print("*** cpuで計算します ***")

    # ハイパーパラメータの設定
    # TODO: jsonに移植
    TEST_RUN = False
    PSE_DATA = False
    HAS_INCEPTION = True
    IS_PREVIOUS = False
    IS_NORMAL = True
    IS_ENN = False  # FIXME: always true so remove here
    IS_MUL_LAYER = False
    CATCH_NREM2 = True
    HAS_DROPOUT = True
    BATCH_SIZE = 512
    N_CLASS = 5
    # KERNEL_SIZE = 256
    KERNEL_SIZE = 128
    STRIDE = 16
    SAMPLE_SIZE = 2000
    UNC_THRETHOLD = 0.3
    DROPOUT_RATE = 0.3
    IS_UNDER_4HZ = False
    DATA_TYPE = "spectrogram"
    FIT_POS = "middle"
    EXPERIMENT_TYPES = (
        "no_cleansing",
        "positive_cleansing",
        "negative_cleansing",
    )
    EXPERIENT_TYPE = "positive_cleansing"
    NORMAL_TAG = "normal" if IS_NORMAL else "sas"
    PSE_DATA_TAG = "psedata" if PSE_DATA else "sleepdata"
    INCEPTION_TAG = "inception" if HAS_INCEPTION else "no-inception"
    ENN_TAG = "enn" if IS_ENN else "dnn"
    INCEPTION_TAG += "v2" if IS_MUL_LAYER else ""
    CATCH_NREM2_TAG = "catch_nrem2" if CATCH_NREM2 else "catch_nrem34"
    CLEANSING_TYPE = "no_cleansing"

    # オブジェクトの作成
    pre_process = PreProcess(
        data_type=DATA_TYPE,
        fit_pos=FIT_POS,
        verbose=0,
        kernel_size=KERNEL_SIZE,
        is_previous=IS_PREVIOUS,
        stride=STRIDE,
        is_normal=IS_NORMAL,
        has_nrem2_bias=False,
        has_rem_bias=False,
        model_type=ENN_TAG,
        cleansing_type=CLEANSING_TYPE,
        make_valdata=True,
        has_ignored=True,
        lsp_option="nr2",
    )
    datasets = pre_process.load_sleep_data.load_data(
        load_all=True, pse_data=PSE_DATA
    )

    # 読み込むモデルの日付リストを返す
    MI = pre_process.my_env.mi
    model_date_d = MI.get_ppi()
    model_date_list = model_date_d[CLEANSING_TYPE]

    # モデルのidを記録するためのリスト
    date_id_saving_list = list()

    for (test_id, test_name), date_id in zip(
        enumerate(pre_process.name_list), model_date_list
    ):
        (train, test) = pre_process.split_train_test_from_records(
            datasets, test_id=test_id, pse_data=PSE_DATA
        )

        # 保存用の時間id
        saving_date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        date_id_saving_list.append(saving_date_id)

        # tagの設定
        # FIXME: name をコード名にする
        main(
            train=train,
            test=test,
            pre_process=pre_process,
            date_id=date_id,
            pse_data=PSE_DATA,
            test_name=test_name,
            n_class=N_CLASS,
            sample_size=SAMPLE_SIZE,
            batch_size=BATCH_SIZE,
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
            is_under_4hz=IS_UNDER_4HZ,
        )

        # testの時は一人の被験者で止める
        if TEST_RUN:
            print(PyColor.RED_FLASH, "testランのため終了します", PyColor.END)
            break
