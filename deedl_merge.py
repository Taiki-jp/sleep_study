import datetime
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from rich import print
from tensorflow.keras.metrics import (
    BinaryAccuracy,  # TrueNegatives,  # TN; TruePositives,  # TP; FalseNegatives,  # FN; FalsePositives,  # FP; 一致率
)
from tensorflow.keras.metrics import Precision  # Precision
from tensorflow.keras.metrics import Recall  # Recall

from data_analysis.py_color import PyColor
from data_analysis.utils import Utils
from nn.losses import EDLLoss
from nn.model_base import EDLModelBase, edl_classifier_1d, edl_classifier_2d
from pre_process.main_param_reader import MainParamReader
from pre_process.pre_process import PreProcess, Record
from pre_process.utils import set_seed


def pred_ss_based_on_unc(
    series: pd.Series,
    counter: Dict[str, int],
) -> Tuple[str, int, int, int, int, int, int, int, int]:
    y_true_bin, bin_pred, unc_bin = (
        series[
            [
                "y_true_nr3",
                "y_true_nr2",
                "y_true_nr1",
                "y_true_rem",
                "y_true_wake",
            ]
        ],
        series[
            [
                "y_pred_nr3",
                "y_pred_nr2",
                "y_pred_nr1",
                "y_pred_rem",
                "y_pred_wake",
            ]
        ],
        series[
            [
                "unc_nr3",
                "unc_nr2",
                "unc_nr1",
                "unc_rem",
                "unc_wake",
            ]
        ],
    )
    # y_trueを5段階に変換
    y_true_bin = y_true_bin.to_numpy()
    y_true = np.argmax(y_true_bin)
    bin_pred = bin_pred.to_numpy()
    # 予測のどれかがpositiveで、ターゲットクラスの予測がpositiveであれば1
    is_included = 1 if sum(bin_pred) != 0 and bin_pred[y_true] == 1 else 0

    # 一つ目のルール
    if sum(bin_pred) == 1:
        counter["rule_0"] += 1
        return (
            1,
            0,
            0,
            np.argmax(bin_pred),
            y_true,
            sum(bin_pred),
            is_included,
        )

    # 二つ目のルール
    elif sum(bin_pred) > 1:
        counter["rule_1"] += 1
        flag_unc = unc_bin
        # フラグが立っている場所だけ残す
        flag_unc = flag_unc.to_numpy() * bin_pred
        # 不確実性が最小の睡眠段階を抽出する
        #             print(flag_unc == np.min(flag_unc[np.nonzero(flag)]))
        #             print(np.argmax(flag_unc == np.min(flag_unc[np.nonzero(flag)])))
        return (
            0,
            1,
            0,
            np.argmax(flag_unc == np.min(flag_unc[np.nonzero(bin_pred)])),
            y_true,
            sum(bin_pred),
            is_included,
        )

    # 三つ目のルール
    else:
        counter["rule_2"] += 1
        assert sum(bin_pred) == 0
        # NR2を返す
        return 0, 0, 1, 1, y_true, sum(bin_pred), is_included


def main(df: pd.DataFrame, testname: str) -> None:
    # y_true
    count_d = {
        "rule_0": 0,
        "rule_1": 0,
        "rule_2": 0,
    }
    output_df = df.apply(
        pred_ss_based_on_unc, axis=1, args=(count_d,), result_type="expand"
    )
    column_names = [
        "rule_0",
        "rule_1",
        "rule_2",
        "y_pred",
        "y_true",
        "up_flags",
        "truth_included",
    ]
    output_df = output_df.rename(
        columns={key: val for key, val in enumerate(column_names)}
    )
    # 一致率の計算
    acc = sum(output_df["y_pred"] == output_df["y_true"]) / output_df.shape[0]
    output_dir = "deedl"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_df.to_csv(os.path.join(output_dir, f"{test_name}.csv"))
    return acc


if __name__ == "__main__":
    # edlとdedlのディレクトリのパス
    target_filename_list = [
        "nr1.csv",
        "nr2.csv",
        "nr3.csv",
        "rem.csv",
        "wake.csv",
        "nr1_double.csv",
        "nr2_double.csv",
        "nr3_double.csv",
        "rem_double.csv",
        "wake_double.csv",
    ]
    input_dir = os.path.join(os.environ["sleep"], "tmps")
    acc_d = dict()
    for dirpath, _, filelist in os.walk(input_dir):
        # 全て同会層にあるので"nr1.csvで引っ掛ける
        if target_filename_list[0] in filelist:
            test_name = os.path.split(dirpath)[1]
            cols = ["y_true", "y_pred", "unc"]
            df = None
            for target_file in target_filename_list:
                names = [
                    __cols + "_" + target_file.split(".")[0] for __cols in cols
                ]

                filepath = os.path.join(dirpath, target_file)
                print(filepath)
                tmp_df = pd.read_csv(
                    os.path.join(dirpath, target_file), usecols=cols
                )
                tmp_df = tmp_df.rename(
                    columns={key: val for key, val in zip(cols, names)}
                )
                if tmp_df is None:
                    df = tmp_df
                else:
                    df = pd.concat([df, tmp_df], axis=1)
            df.to_csv("tmp.csv")
            acc = main(df, test_name)
            acc_d.update({test_name: [acc]})
    print(acc_d)
    acc_df = pd.DataFrame.from_dict(acc_d)
    acc_df.to_csv("deedl_acc.csv")
