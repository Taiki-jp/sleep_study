# 2クラス分類の結果をマージする
import os
import sys
from glob import glob
from multiprocessing import Pool
from tkinter import W
from typing import Dict, List, Tuple, Union
from unittest import result

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from psutil import cpu_count
from tqdm import tqdm


# ランダム選択の予測確率を計算する(up_flagsが2以上の時にtruth_includedが1の行の確率の和を取る)
def calc_random_select_acc(
    df: pd.DataFrame, case: str
) -> Union[
    Tuple[float, float, float], Tuple[float, float, float, float, float]
]:
    div = lambda series: np.array(1, dtype="float") / np.array(
        series["up_flags"], dtype="float"
    )
    random_acc = df.apply(div, axis=1)
    # infをnanに置換
    random_acc = random_acc.replace([np.inf, -np.inf], np.nan)
    if case == "eenn":
        exploit_condition_rule_0 = np.logical_and(
            df["rule_0"] == 1, df["truth_included"]
        )
        exploit_condition_rule_1 = np.logical_and(
            df["rule_1"] == 1, df["truth_included"]
        )
        return (
            random_acc[exploit_condition_rule_0].mean(),
            random_acc[exploit_condition_rule_1].mean(),
            1 / 5,
        )
    else:
        exploit_condition_rule_0 = np.logical_and(
            df["rule_0"] == 1, df["truth_included"]
        )
        exploit_condition_rule_1 = np.logical_and(
            df["rule_1"] == 1, df["truth_included"]
        )
        exploit_condition_rule_2 = np.logical_and(
            df["rule_2"] == 1, df["truth_included"]
        )
        exploit_condition_rule_3 = np.logical_and(
            df["rule_3"] == 1, df["truth_included"]
        )
        return (
            random_acc[exploit_condition_rule_0].mean(),
            random_acc[exploit_condition_rule_1].mean(),
            random_acc[exploit_condition_rule_2].mean(),
            random_acc[exploit_condition_rule_3].mean(),
            1 / 5,
        )


# それぞれのルールの予測の正解率を計算する(予測があっていた数/予測された総和)
# FIXME: aecnnのルール0の値がinfのせい？その後の処理によりnanになっている
# FIXME: eennのルール2の確率の計算方法を変更する
def calc_rulebase_select_acc(
    df: pd.DataFrame, case: str
) -> Union[
    Tuple[float, float, float], Tuple[float, float, float, float, float]
]:
    div = lambda x, y: np.array(x, dtype="float") / np.array(y, dtype="float")
    if case == "eenn":
        # 総和（各ルールが選択されてその予測の一つが正解の場合の数）
        exploit_condition_rule_0 = np.logical_and(
            df["rule_0"] == 1, df["truth_included"]
        )
        exploit_condition_rule_1 = np.logical_and(
            df["rule_1"] == 1, df["truth_included"]
        )
        exploit_condition_rule_2 = np.logical_and(
            df["rule_2"] == 1, df["truth_included"]
        )
        rule_0_selected_num = df["rule_0"][exploit_condition_rule_0].sum()
        rule_1_selected_num = df["rule_1"][exploit_condition_rule_1].sum()
        rule_2_selected_num = df["rule_2"][exploit_condition_rule_2].sum()
        # 各ルールが正解の数
        flagged_rule_0 = df[df["rule_0"] == 1]
        flagged_rule_1 = df[df["rule_1"] == 1]
        flagged_rule_2 = df[df["rule_2"] == 1]
        rule_0_correct_num = flagged_rule_0["rule_0"][
            flagged_rule_0["y_pred"] == flagged_rule_0["y_true"]
        ].sum()
        rule_1_correct_num = flagged_rule_1["rule_1"][
            flagged_rule_1["y_pred"] == flagged_rule_1["y_true"]
        ].sum()
        rule_2_correct_num = flagged_rule_2["rule_2"][
            flagged_rule_2["y_pred"] == flagged_rule_2["y_true"]
        ].sum()
        return tuple(
            map(
                div,
                [rule_0_correct_num, rule_1_correct_num, 1],
                [
                    rule_0_selected_num,
                    rule_1_selected_num,
                    5,
                ],
            )
        )
    else:
        # 総和（各ルールが選択されてその予測の一つが正解の場合の数）
        exploit_condition_rule_0 = np.logical_and(
            df["rule_0"] == 1, df["truth_included"]
        )
        exploit_condition_rule_1 = np.logical_and(
            df["rule_1"] == 1, df["truth_included"]
        )
        exploit_condition_rule_2 = np.logical_and(
            df["rule_2"] == 1, df["truth_included"]
        )
        exploit_condition_rule_3 = np.logical_and(
            df["rule_3"] == 1, df["truth_included"]
        )
        exploit_condition_rule_4 = np.logical_and(
            df["rule_4"] == 1, df["truth_included"]
        )
        # 上の条件に引っかかる行を抽出
        rule_0_selected_num = df["rule_0"][exploit_condition_rule_0].sum()
        rule_1_selected_num = df["rule_1"][exploit_condition_rule_1].sum()
        rule_2_selected_num = df["rule_2"][exploit_condition_rule_2].sum()
        rule_3_selected_num = df["rule_3"][exploit_condition_rule_3].sum()
        flagged_rule_0 = df[exploit_condition_rule_0]
        flagged_rule_1 = df[exploit_condition_rule_1]
        flagged_rule_2 = df[exploit_condition_rule_2]
        flagged_rule_3 = df[exploit_condition_rule_3]
        # 各ルールが正解の数
        rule_0_correct_num = flagged_rule_0["rule_0"][
            flagged_rule_0["y_pred"] == flagged_rule_0["y_true"]
        ].sum()
        rule_1_correct_num = flagged_rule_1["rule_1"][
            flagged_rule_1["y_pred"] == flagged_rule_1["y_true"]
        ].sum()
        rule_2_correct_num = flagged_rule_2["rule_2"][
            flagged_rule_2["y_pred"] == flagged_rule_2["y_true"]
        ].sum()
        rule_3_correct_num = flagged_rule_3["rule_3"][
            flagged_rule_3["y_pred"] == flagged_rule_3["y_true"]
        ].sum()
        return tuple(
            map(
                div,
                [
                    rule_0_correct_num,
                    rule_1_correct_num,
                    rule_2_correct_num,
                    rule_3_correct_num,
                    1,
                ],
                [
                    rule_0_selected_num,
                    rule_1_selected_num,
                    rule_2_selected_num,
                    rule_3_selected_num,
                    1,
                ],
            )
        )


def main(
    aecnn_filelist: List[str],
    ecnn_filelist: List[str],
    ccnn_filelist: List[str],
    eenn_filelist: List[str],
):

    aecnn_random_accs_list = list()
    ecnn_random_accs_list = list()
    ccnn_random_accs_list = list()
    eenn_random_accs_list = list()
    aecnn_rulebase_accs_list = list()
    ecnn_rulebase_accs_list = list()
    ccnn_rulebase_accs_list = list()
    eenn_rulebase_accs_list = list()

    for aecnn_file, ecnn_file, ccnn_file, eenn_file in zip(
        aecnn_filelist, ecnn_filelist, ccnn_filelist, eenn_filelist
    ):
        aecnn_df, ecnn_df, ccnn_df, eenn_df = map(
            pd.read_csv, [aecnn_file, ecnn_file, ccnn_file, eenn_file]
        )
        # ランダム選択の予測確率を計算する(up_flagsが2以上の時にtruth_includedが1の行の確率の和を取る)
        aecnn_random_accs = calc_random_select_acc(df=aecnn_df, case="aecnn")
        ecnn_random_accs = calc_random_select_acc(df=ecnn_df, case="ecnn")
        ccnn_random_accs = calc_random_select_acc(df=ccnn_df, case="ccnn")
        eenn_random_accs = calc_random_select_acc(df=eenn_df, case="eenn")
        # それぞれのルールの予測の正解率を計算する(予測があっていた数/予測された総和)
        aecnn_rulebase_accs = calc_rulebase_select_acc(
            df=aecnn_df, case="aecnn"
        )
        ecnn_rulebase_accs = calc_rulebase_select_acc(df=ecnn_df, case="ecnn")
        ccnn_rulebase_accs = calc_rulebase_select_acc(df=ccnn_df, case="ccnn")
        eenn_rulebase_accs = calc_rulebase_select_acc(df=eenn_df, case="eenn")
        # リストに追加
        aecnn_random_accs_list.append(list(aecnn_random_accs))
        ecnn_random_accs_list.append(list(ecnn_random_accs))
        ccnn_random_accs_list.append(list(ccnn_random_accs))
        eenn_random_accs_list.append(list(eenn_random_accs))
        aecnn_rulebase_accs_list.append(list(aecnn_rulebase_accs))
        ecnn_rulebase_accs_list.append(list(ecnn_rulebase_accs))
        ccnn_rulebase_accs_list.append(list(ccnn_rulebase_accs))
        eenn_rulebase_accs_list.append(list(eenn_rulebase_accs))
    # 各手法のランダム選択とルールベース選択をルール別に一致率を計算
    # aecnn
    aecnn_random_accs_rule_0 = [arr[0] for arr in aecnn_random_accs_list]
    aecnn_rulebase_accs_rule_0 = [arr[0] for arr in aecnn_rulebase_accs_list]
    aecnn_random_accs_rule_1 = [arr[1] for arr in aecnn_random_accs_list]
    aecnn_rulebase_accs_rule_1 = [arr[1] for arr in aecnn_rulebase_accs_list]
    aecnn_random_accs_rule_2 = [arr[2] for arr in aecnn_random_accs_list]
    aecnn_rulebase_accs_rule_2 = [arr[2] for arr in aecnn_rulebase_accs_list]
    aecnn_random_accs_rule_3 = [arr[3] for arr in aecnn_random_accs_list]
    aecnn_rulebase_accs_rule_3 = [arr[3] for arr in aecnn_rulebase_accs_list]
    aecnn_random_accs_rule_4 = [arr[4] for arr in aecnn_random_accs_list]
    aecnn_rulebase_accs_rule_4 = [arr[4] for arr in aecnn_rulebase_accs_list]
    # ecnn
    ecnn_random_accs_rule_0 = [arr[0] for arr in ecnn_random_accs_list]
    ecnn_rulebase_accs_rule_0 = [arr[0] for arr in ecnn_rulebase_accs_list]
    ecnn_random_accs_rule_1 = [arr[1] for arr in ecnn_random_accs_list]
    ecnn_rulebase_accs_rule_1 = [arr[1] for arr in ecnn_rulebase_accs_list]
    ecnn_random_accs_rule_2 = [arr[2] for arr in ecnn_random_accs_list]
    ecnn_rulebase_accs_rule_2 = [arr[2] for arr in ecnn_rulebase_accs_list]
    ecnn_random_accs_rule_3 = [arr[3] for arr in ecnn_random_accs_list]
    ecnn_rulebase_accs_rule_3 = [arr[3] for arr in ecnn_rulebase_accs_list]
    ecnn_random_accs_rule_4 = [arr[4] for arr in ecnn_random_accs_list]
    ecnn_rulebase_accs_rule_4 = [arr[4] for arr in ecnn_rulebase_accs_list]
    # ccnn
    ccnn_random_accs_rule_0 = [arr[0] for arr in ccnn_random_accs_list]
    ccnn_rulebase_accs_rule_0 = [arr[0] for arr in ccnn_rulebase_accs_list]
    ccnn_random_accs_rule_1 = [arr[1] for arr in ccnn_random_accs_list]
    ccnn_rulebase_accs_rule_1 = [arr[1] for arr in ccnn_rulebase_accs_list]
    ccnn_random_accs_rule_2 = [arr[2] for arr in ccnn_random_accs_list]
    ccnn_rulebase_accs_rule_2 = [arr[2] for arr in ccnn_rulebase_accs_list]
    ccnn_random_accs_rule_3 = [arr[3] for arr in ccnn_random_accs_list]
    ccnn_rulebase_accs_rule_3 = [arr[3] for arr in ccnn_rulebase_accs_list]
    ccnn_random_accs_rule_4 = [arr[4] for arr in ccnn_random_accs_list]
    ccnn_rulebase_accs_rule_4 = [arr[4] for arr in ccnn_rulebase_accs_list]
    # eenn
    eenn_random_accs_rule_0 = [arr[0] for arr in eenn_random_accs_list]
    eenn_rulebase_accs_rule_0 = [arr[0] for arr in eenn_rulebase_accs_list]
    eenn_random_accs_rule_1 = [arr[1] for arr in eenn_random_accs_list]
    eenn_rulebase_accs_rule_1 = [arr[1] for arr in eenn_rulebase_accs_list]
    eenn_random_accs_rule_2 = [arr[2] for arr in eenn_random_accs_list]
    eenn_rulebase_accs_rule_2 = [arr[2] for arr in eenn_rulebase_accs_list]
    column_name = (
        "aecnn_random_rule_0",
        "aecnn_rulebase_rule_0",
        "aecnn_random_rule_1",
        "aecnn_rulebase_rule_1",
        "aecnn_random_rule_2",
        "aecnn_rulebase_rule_2",
        "aecnn_random_rule_3",
        "aecnn_rulebase_rule_3",
        "aecnn_random_rule_4",
        "aecnn_rulebase_rule_4",
        "ecnn_random_rule_0",
        "ecnn_rulebase_rule_0",
        "ecnn_random_rule_1",
        "ecnn_rulebase_rule_1",
        "ecnn_random_rule_2",
        "ecnn_rulebase_rule_2",
        "ecnn_random_rule_3",
        "ecnn_rulebase_rule_3",
        "ecnn_random_rule_4",
        "ecnn_rulebase_rule_4",
        "ccnn_random_rule_0",
        "ccnn_rulebase_rule_0",
        "ccnn_random_rule_1",
        "ccnn_rulebase_rule_1",
        "ccnn_random_rule_2",
        "ccnn_rulebase_rule_2",
        "ccnn_random_rule_3",
        "ccnn_rulebase_rule_3",
        "ccnn_random_rule_4",
        "ccnn_rulebase_rule_4",
        "eenn_random_rule_0",
        "eenn_rulebase_rule_0",
        "eenn_random_rule_1",
        "eenn_rulebase_rule_1",
        "eenn_random_rule_2",
        "eenn_rulebase_rule_2",
    )
    d4df = {
        key: val
        for key, val in zip(
            column_name,
            [
                aecnn_random_accs_rule_0,
                aecnn_rulebase_accs_rule_0,
                aecnn_random_accs_rule_1,
                aecnn_rulebase_accs_rule_1,
                aecnn_random_accs_rule_2,
                aecnn_rulebase_accs_rule_2,
                aecnn_random_accs_rule_3,
                aecnn_rulebase_accs_rule_3,
                aecnn_random_accs_rule_4,
                aecnn_rulebase_accs_rule_4,
                ecnn_random_accs_rule_0,
                ecnn_rulebase_accs_rule_0,
                ecnn_random_accs_rule_1,
                ecnn_rulebase_accs_rule_1,
                ecnn_random_accs_rule_2,
                ecnn_rulebase_accs_rule_2,
                ecnn_random_accs_rule_3,
                ecnn_rulebase_accs_rule_3,
                ecnn_random_accs_rule_4,
                ecnn_rulebase_accs_rule_4,
                ccnn_random_accs_rule_0,
                ccnn_rulebase_accs_rule_0,
                ccnn_random_accs_rule_1,
                ccnn_rulebase_accs_rule_1,
                ccnn_random_accs_rule_2,
                ccnn_rulebase_accs_rule_2,
                ccnn_random_accs_rule_3,
                ccnn_rulebase_accs_rule_3,
                ccnn_random_accs_rule_4,
                ccnn_rulebase_accs_rule_4,
                eenn_random_accs_rule_0,
                eenn_rulebase_accs_rule_0,
                eenn_random_accs_rule_1,
                eenn_rulebase_accs_rule_1,
                eenn_random_accs_rule_2,
                eenn_rulebase_accs_rule_2,
            ],
        )
    }
    df = pd.DataFrame.from_dict(data=d4df)
    output_file = os.path.join(os.environ["sleep"], "tmps", "shono_ans.csv")
    df.to_csv(output_file)
    return


if __name__ == "__main__":
    # aecnn
    aecnn_filedir = os.path.join(os.environ["sleep"], "logs", "aecnn", "*.csv")
    aecnn_filelist = glob(aecnn_filedir)
    # ecnn
    ecnn_filedir = os.path.join(os.environ["sleep"], "logs", "ecnn", "*.csv")
    ecnn_filelist = glob(ecnn_filedir)
    # ccnn
    ccnn_filedir = os.path.join(os.environ["sleep"], "logs", "ccnn", "*.csv")
    ccnn_filelist = glob(ccnn_filedir)
    # eenn
    eenn_filedir = os.path.join(os.environ["sleep"], "logs", "eenn", "*.csv")
    eenn_filelist = glob(eenn_filedir)
    main(aecnn_filelist, ecnn_filelist, ccnn_filelist, eenn_filelist)
