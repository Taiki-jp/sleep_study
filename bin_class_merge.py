import datetime
import os
import sys
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops.numpy_ops.np_math_ops import positive

from data_analysis.py_color import PyColor
from data_analysis.utils import Utils
from nn.utils import load_bin_model, separate_unc_data
from pre_process.json_base import JsonBase
from pre_process.pre_process import PreProcess
from pre_process.utils import set_seed


def main(
    is_enn: bool,
    pse_data: bool,
    target_ss: str,
    is_under_4hz: bool,
    name: str,
    train: list,
    test: list,
    pre_process: PreProcess,
    utils: Utils,
    my_tags: list = None,
    n_class: int = 5,
    test_name: str = None,
    date_id: dict = dict(),
    sample_size: int = 0,
    batch_size: int = 0,
    unc_threthold: float = 0,
    log_all_in_one: bool = False,
):

    # データセットの作成
    ((_), (_), (x_test, y_test),) = pre_process.make_dataset(
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

    # 5つのモデルでevidenceを出力
    evd_list = list()
    alp_list = list()
    unc_list = list()
    y_pred_list = list()
    for _model in model:
        # ennの時はevidenceから計算する
        if is_enn:
            _evd = _model.predict(x_test, batch_size=batch_size)
            evd_list.append(_evd)
            _alp, _, _unc, _y_pred = utils.calc_enn_output_from_evidence(
                evidence=_evd
            )
            alp_list.append(_alp)
            unc_list.append(_unc)
            y_pred_list.append(_y_pred)
        # dnnの時は直接確率を計算する
        else:
            y_pred = _model.predict(x_test, batch_size=batch_size)
            y_pred_list.append(np.argmax(y_pred, axis=1).tolist())

    if is_enn:
        # y_test, y_pred, uncの順に結合する
        y_pred_list.extend(unc_list)
    y_pred_list.append(list(y_test[0]))
    # 時刻を取り出す
    time_list = [__time.time for __time in test]
    # 先頭に時刻を挿入
    y_pred_list.insert(0, time_list)
    # 行列方向を入れ替える
    if is_enn:
        output_df = pd.DataFrame(
            y_pred_list,
            index=[
                "time",
                "nr1_pred",
                "nr2_pred",
                "nr34_pred",
                "rem_pred",
                "wake_pred",
                "nr1_unc",
                "nr2_unc",
                "nr34_unc",
                "rem_unc",
                "wake_unc",
                "y_true",
            ],
        ).T
    else:
        output_df = pd.DataFrame(
            y_pred_list,
            index=[
                "time",
                "nr1_pred",
                "nr2_pred",
                "nr34_pred",
                "rem_pred",
                "wake_pred",
                "y_true",
            ],
        ).T

    try:
        assert test_name is not None
    except AssertionError("testname is none"):
        sys.exit(1)
    filepath = os.path.join(
        os.environ["sleep"],
        "log",
        "bin_merge",
        "1_1",
        f"output_{test_name}.csv",
    )
    # path check
    filedir, _ = os.path.split(filepath)
    if not os.path.exists(filedir):
        print(PyColor.RED_FLASH, f"filedir:{filedir}を作成します", PyColor.END)
        os.makedirs(filedir)
    # show filename
    print(PyColor.RED_FLASH, f"saving {filepath} ...", PyColor.END)
    output_df.to_csv(filepath)

    # # ベースモデルの不確実なデータセットに対する一致率を計算しwandbに送信
    # for base_or_positive, _model in zip(
    #     ("base", "positive"), (model, positive_model)
    # ):
    #     Utils().calc_ss_acc(
    #         x=_x_test,
    #         y=_y_test,
    #         model=_model,
    #         n_class=n_class,
    #         batch_size=batch_size,
    #         base_or_positive=base_or_positive,
    #     )

    # # # クレンジング後のデータに対してグラフを作成
    # # 不確実性の高いデータのみで一致率を計算
    # evidence_base = model.predict(_x_test, batch_size=batch_size)
    # evidence_positive = positive_model.predict(_x_test)

    # # 睡眠段階の予測
    # _, _, _, y_pred_base = utils.calc_enn_output_from_evidence(
    #     evidence=evidence_base
    # )
    # _, _, _, y_pred_pos = utils.calc_enn_output_from_evidence(
    #     evidence=evidence_positive
    # )
    # # 一致率の計算
    # # acc_base = utils.calc_acc_from_pred(
    # #     y_true=_y_test.numpy(), y_pred=y_pred_base, log_label="base"
    # # )
    # # acc_sub = utils.calc_acc_from_pred(
    # #     y_true=_y_test.numpy(), y_pred=y_pred_pos, log_label="sub"
    # # )
    # # # csv出力
    # # output_path = "20211018_for_box_plot.csv"
    # # utils.to_csv(df_result, path=output_path, edit_mode="append")


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
    TEST_RUN = False
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
    SAMPLE_SIZE = 2000
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
        date_id = model_date_list[test_name]
        # 2クラス分類の時はこれで作ってしまっているのでこれに合わせるしかない
        (train, test) = pre_process.split_train_test_from_records(
            datasets, test_id=test_id, pse_data=PSE_DATA
        )

        # tagの設定
        my_tags = [
            test_name,
        ]
        # FIXME: name をコード名にする
        main(
            is_enn=IS_ENN,
            name=f"{test_name}",
            train=train,
            test=test,
            pre_process=pre_process,
            my_tags=my_tags,
            date_id=date_id,
            test_name=test_name,
            n_class=N_CLASS,
            batch_size=BATCH_SIZE,
            unc_threthold=UNC_THRETHOLD,
            log_all_in_one=True,
            sample_size=SAMPLE_SIZE,  # これがないとtrainの分割のところでデータがなくてエラーが起きてしまう
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
            pse_data=PSE_DATA,
            target_ss="wake",
            is_under_4hz=IS_UNDER_4HZ,
        )

        # testの時は一人の被験者で止める
        if TEST_RUN:
            print(PyColor.RED_FLASH, "testランのため終了します", PyColor.END)
            break
