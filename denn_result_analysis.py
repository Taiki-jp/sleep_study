import datetime
import os
import sys
from collections import Counter
from glob import glob
from typing import Any, Dict, List, Tuple

import pandas as pd
import tensorflow as tf
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


def enn_main(df: pd.DataFrame, utils: Utils, output_filepath: str):
    select = (
        lambda series: series["base_pred"]
        if series["unc_base"] < 0.3
        else series["positive_pred"]
    )
    # dennの予測
    y_pred = df.apply(select, axis=1)
    # ennの予測とdennの予測を比較するグラフを作成する
    utils.make_graphs_comparing_enn_denn(df, y_pred, output_filepath)
    return


def cnn_main(
    df_5stage: pd.DataFrame,
    df_bin: pd.DataFrame,
    utils: Utils,
    output_filepath: str,
):
    # ennの予測とdennの予測を比較するグラフを作成する
    utils.make_graphs_comparing_cnn(df_5stage, df_bin, output_filepath)
    return


if __name__ == "__main__":
    utils = Utils(
        is_normal=True,
        is_previous=False,
        data_type="",
        fit_pos="",
        stride=16,
        kernel_size=128,
        model_type="",
        cleansing_type="",
    )
    # ennの入力ディレクトリ
    enn_input_dir = os.path.join(os.environ["git"], "sleep_study", "nidan_enn")
    # cnn-5段階同時の入力ディレクトリ
    cnn_5stage_input_dir = os.path.join(
        os.environ["git"], "sleep_study", "cnn_attn_5stage"
    )
    cnn_bin_input_dir = os.path.join(os.environ["sleep"], "logs", "aecnn")
    # cnn-5段階同時（アテンションなし）の入力ディレクトリ
    cnn_5stage_input_dir = os.path.join(
        os.environ["git"], "sleep_study", "cnn_noattn_5stage"
    )
    cnn_bin_input_dir = os.path.join(os.environ["sleep"], "logs", "ecnn")

    # ennのグラフ作成
    # for dir, _, target_file in os.walk(enn_input_dir):
    #     if "ss_and_unc.csv" in target_file:
    #         df = pd.read_csv(os.path.join(dir, "ss_and_unc.csv"))
    #         testname = os.path.split(dir)[1]
    #         output_filepath = os.path.join(
    #             os.environ["git"],
    #             "sleep_study",
    #             "enn_comparison",
    #             f"{testname}.png",
    #         )
    #         if not os.path.exists(os.path.split(output_filepath)[0]):
    #             os.makedirs(os.path.split(output_filepath)[0])
    #         enn_main(df, utils, output_filepath)
    #     else:
    #         continue

    # # cnnのグラフ作成
    # # 同時推定の場合はファイル名一致するもののみをリストに追加する
    # cnn_5stage_list = list()
    # testname_list = list()
    # for (cnn_5stage_dir, _, cnn_5stage_dir_target_file) in os.walk(
    #     cnn_5stage_input_dir
    # ):
    #     if "ss_5class.csv" in cnn_5stage_dir_target_file:
    #         cnn_5stage_list.append(
    #             os.path.join(cnn_5stage_dir, "ss_5class.csv")
    #         )
    #         testname_list.append(os.path.split(cnn_5stage_dir)[1])
    # # binファイルは全て拾う
    # cnn_bin_list = glob(os.path.join(cnn_bin_input_dir, "*"))
    # # 長さが同じことを最低限確認
    # assert len(cnn_5stage_list) == len(cnn_bin_list)
    # for five_path, bin_path, testname in zip(
    #     cnn_5stage_list, cnn_bin_list, testname_list
    # ):
    #     df_cnn_5stage = pd.read_csv(five_path)
    #     df_cnn_bin = pd.read_csv(bin_path)
    #     output_filepath = os.path.join(
    #         os.environ["git"],
    #         "sleep_study",
    #         "cnn_comparison",
    #         f"{testname}.png",
    #     )
    #     if not os.path.exists(os.path.split(output_filepath)[0]):
    #         os.makedirs(os.path.split(output_filepath)[0])
    #     cnn_main(df_cnn_5stage, df_cnn_bin, utils, output_filepath)

    # cnn（アテンションなし）のグラフ作成
    # 同時推定の場合はファイル名一致するもののみをリストに追加する
    cnn_noattn_5stage_list = list()
    testname_list = list()
    for (cnn_5stage_dir, _, cnn_5stage_dir_target_file) in os.walk(
        cnn_5stage_input_dir
    ):
        if "ss_5class.csv" in cnn_5stage_dir_target_file:
            cnn_noattn_5stage_list.append(
                os.path.join(cnn_5stage_dir, "ss_5class.csv")
            )
            testname_list.append(os.path.split(cnn_5stage_dir)[1])
    # binファイルは全て拾う
    cnn_bin_list = glob(os.path.join(cnn_bin_input_dir, "*"))
    # 長さが同じことを最低限確認
    assert len(cnn_noattn_5stage_list) == len(cnn_bin_list)
    for five_path, bin_path, testname in zip(
        cnn_noattn_5stage_list, cnn_bin_list, testname_list
    ):
        df_cnn_5stage = pd.read_csv(five_path)
        df_cnn_bin = pd.read_csv(bin_path)
        output_filepath = os.path.join(
            os.environ["git"],
            "sleep_study",
            "cnn_no_attn_comparison",
            f"{testname}.png",
        )
        if not os.path.exists(os.path.split(output_filepath)[0]):
            os.makedirs(os.path.split(output_filepath)[0])
        cnn_main(df_cnn_5stage, df_cnn_bin, utils, output_filepath)
