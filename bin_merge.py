# 2クラス分類の結果をマージする
import os
import sys
from glob import glob
from multiprocessing import Pool
from typing import Dict, Tuple
from unittest import result

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from psutil import cpu_count
from tqdm import tqdm

# ennのパス(eenn)
enn_filedir = os.path.join(
    os.environ["git"], "sleep_study", "bin_enn", "*.csv"
)
# cnn+attnのパス(aecnn)
cnn_attn_filedir = os.path.join(
    os.environ["git"], "sleep_study", "cnn_attn", "*.csv"
)
# cnnのパス(ecnn)
cnn_filedir = os.path.join(
    os.environ["git"], "sleep_study", "cnn_no_attn", "*.csv"
)

enn_filelist = glob(enn_filedir)
cnn_attn_filelist = glob(cnn_attn_filedir)
cnn_filelist = glob(cnn_filedir)


def pred_ss_based_on_unc(
    series: pd.Series,
    counter: Dict[str, int],
) -> Tuple[str, int, int, int, int, int, int, int, int]:
    time, y_true, bin_pred = (
        series["time"],
        series["y_true"],
        series[["nr34_pred", "nr2_pred", "nr1_pred", "rem_pred", "wake_pred"]],
    )
    bin_pred = bin_pred.to_numpy()
    is_included = (
        1 if sum(bin_pred) != 0 and np.argmax(bin_pred) == y_true else 0
    )

    # 一つ目のルール
    if sum(bin_pred) == 1:
        counter["rule_0"] += 1
        return (
            time,
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
        flag_unc = series[
            ["nr34_unc", "nr2_unc", "nr1_unc", "rem_unc", "wake_unc"]
        ]
        # フラグが立っている場所だけ残す
        flag_unc = flag_unc.to_numpy() * bin_pred
        # 不確実性が最小の睡眠段階を抽出する
        #             print(flag_unc == np.min(flag_unc[np.nonzero(flag)]))
        #             print(np.argmax(flag_unc == np.min(flag_unc[np.nonzero(flag)])))
        return (
            time,
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
        return time, 0, 0, 1, 1, y_true, sum(bin_pred), is_included


def pred_ss_based_on_bio(series: pd.Series, counter: Dict[str, int]):
    time, y_true, bin_pred = (
        series["time"],
        series["y_true"],
        series[["nr34_pred", "nr2_pred", "nr1_pred", "rem_pred", "wake_pred"]],
    )
    bin_pred = bin_pred.to_numpy()
    is_included = (
        1 if sum(bin_pred) != 0 and np.argmax(bin_pred) == y_true else 0
    )
    # 0. ひとつだけPositiveならそのクラス
    if sum(bin_pred) == 1:
        counter["rule_0"] += 1
        return (
            time,
            1,
            0,
            0,
            0,
            0,
            np.argmax(bin_pred),
            y_true,
            sum(bin_pred),
            is_included,
        )

    # 1. NR2, remがPositiveならrem
    elif bin_pred[2] == 1 and bin_pred[4] == 1:
        counter["rule_1"] += 1
        return time, 0, 1, 0, 0, 0, 3, y_true, sum(bin_pred), is_included

    # 2. nr1, wakeがpositiveならwake
    elif bin_pred[2] == 1 and bin_pred[4] == 1:
        counter["rule_2"] += 1
        return time, 0, 0, 1, 0, 0, 4, y_true, sum(bin_pred), is_included

    # 3. positiveのなかで優先度の高いもので並び替え
    elif int(sum(bin_pred)) > 1:
        counter["rule_3"] += 1
        if bin_pred[1] == 1:
            y_pred = 1
        elif bin_pred[3] == 1:
            y_pred = 3
        elif bin_pred[2] == 1:
            y_pred = 2
        elif bin_pred[4] == 1:
            y_pred = 4
        elif bin_pred[0] == 1:
            y_pred = 0
        else:
            raise AssertionError
        return time, 0, 0, 0, 1, 0, y_pred, y_true, sum(bin_pred), is_included
    else:
        counter["rule_4"] += 1
        return time, 0, 0, 0, 0, 1, 1, y_true, sum(bin_pred), is_included


def pred_ss_custom(series: pd.Series, counter: Dict[str, int]) -> pd.DataFrame:
    time, y_true = series[["attn_time", "attn_y_true"]]
    bin_pred_attn = series[
        [
            "attn_nr34_pred",
            "attn_nr2_pred",
            "attn_nr1_pred",
            "attn_rem_pred",
            "attn_wake_pred",
        ]
    ]
    bin_pred_no_attn = series[
        [
            "no_attn_nr34_pred",
            "no_attn_nr2_pred",
            "no_attn_nr1_pred",
            "no_attn_rem_pred",
            "no_attn_wake_pred",
        ]
    ]
    # 使用する予測器をカスタム（attn: NR34, REM, no-attn: NR2, NR1, WAKE）
    bin_pred = np.array(
        [
            bin_pred_attn[0],
            bin_pred_no_attn[1],
            bin_pred_no_attn[2],
            bin_pred_attn[3],
            bin_pred_no_attn[4],
        ]
    )
    is_included = (
        1 if sum(bin_pred) != 0 and np.argmax(bin_pred) == y_true else 0
    )
    # 0. ひとつだけPositiveならそのクラス
    if sum(bin_pred) == 1:
        counter["rule_0"] += 1
        return (
            time,
            1,
            0,
            0,
            0,
            0,
            np.argmax(bin_pred),
            y_true,
            sum(bin_pred),
            is_included,
        )

    # 1. NR2, remがPositiveならrem
    elif bin_pred[2] == 1 and bin_pred[4] == 1:
        counter["rule_1"] += 1
        return time, 0, 1, 0, 0, 0, 3, y_true, sum(bin_pred), is_included

    # 2. nr1, wakeがpositiveならwake
    elif bin_pred[2] == 1 and bin_pred[4] == 1:
        counter["rule_2"] += 1
        return time, 0, 0, 1, 0, 0, 4, y_true, sum(bin_pred), is_included

    # 3. positiveのなかで優先度の高いもので並び替え
    elif int(sum(bin_pred)) > 1:
        counter["rule_3"] += 1
        if bin_pred[1] == 1:
            y_pred = 1
        elif bin_pred[3] == 1:
            y_pred = 3
        elif bin_pred[2] == 1:
            y_pred = 2
        elif bin_pred[4] == 1:
            y_pred = 4
        elif bin_pred[0] == 1:
            y_pred = 0
        else:
            raise AssertionError
        return time, 0, 0, 0, 1, 0, y_pred, y_true, sum(bin_pred), is_included
    else:
        counter["rule_4"] += 1
        return time, 0, 0, 0, 0, 1, 1, y_true, sum(bin_pred), is_included


def pred_ss_custom_ver2(
    attn_df: pd.DataFrame, no_attn_df: pd.DataFrame, counter: Dict[str, int]
):
    y_pred = list()
    bin_pred_attn = attn_df[
        ["nr34_pred", "nr2_pred", "nr1_pred", "rem_pred", "wake_pred"]
    ]
    bin_pred_no_attn = no_attn_df[
        ["nr34_pred", "nr2_pred", "nr1_pred", "rem_pred", "wake_pred"]
    ]
    for (_, _bin_pred_attn), (_, _bin_pred_no_attn) in zip(
        bin_pred_attn.iterrows(), bin_pred_no_attn.iterrows()
    ):
        # print(_bin_pred_attn)
        # print(_bin_pred_no_attn)
        # continue
        # 0. ひとつだけPositiveならそのクラス
        array4case_1 = np.array(
            [
                _bin_pred_no_attn[0],
                _bin_pred_no_attn[1],
                _bin_pred_attn[2],
                _bin_pred_attn[3],
                _bin_pred_no_attn[4],
            ]
        )
        if sum(array4case_1) == 1:
            counter["rule_0"] += 1
            y_pred.append(np.argmax(_bin_pred_attn))
        elif (
            _bin_pred_attn[0] == 1
            or _bin_pred_no_attn[1] == 1
            or _bin_pred_attn[3] == 1
        ):
            counter["rule_1"] += 1
            y_pred.append(np.argmax(array4case_1))

        elif (
            _bin_pred_no_attn[2] == 1
            or _bin_pred_attn[4] == 1
            and pd.concat([_bin_pred_no_attn, _bin_pred_attn], axis=0).sum()
            == 2
        ):
            counter["rule_2"] += 1
            y_pred.append(4)
        elif (
            pd.concat([_bin_pred_no_attn, _bin_pred_attn], axis=0).sum() > 1
            and sum(array4case_1) > 1
        ):
            counter["rule_3"] += 1
            if array4case_1[1] == 1:
                y_pred.append(1)
            elif array4case_1[3] == 1:
                y_pred.append(3)
            elif array4case_1[2] == 1:
                y_pred.append(2)
            elif array4case_1[4] == 1:
                y_pred.append(4)
            elif array4case_1[0] == 1:
                y_pred.append(0)
            else:
                raise AssertionError
        else:
            counter["rule_4"] += 1
            y_pred.append(0)
    return pd.DataFrame(y_pred, columns=["y_pred"])


def make_ecnn_table(cnn_file):
    count_d4b = {
        "rule_0": 0,
        "rule_1": 0,
        "rule_2": 0,
        "rule_3": 0,
        "rule_4": 0,
    }
    df = pd.read_csv(cnn_file)
    new_df = df.apply(
        pred_ss_based_on_bio, args=(count_d4b,), axis=1, result_type="expand"
    )
    column_names = [
        "time",
        "rule_0",
        "rule_1",
        "rule_2",
        "rule_3",
        "rule_4",
        "y_pred",
        "y_true",
        "up_flags",
        "truth_included",
    ]
    new_df = new_df.rename(
        columns={key: val for key, val in enumerate(column_names)}
    )
    output_dir = os.path.join(os.environ["sleep"], "logs", "ecnn")
    if not os.path.exists(output_dir):
        print(f"{output_dir}を作成します")
        os.makedirs(output_dir)
    filename = os.path.split(cnn_file)[1]
    # ファイル名が"output"から始まっている場合は取り除く
    if filename.startswith("output"):
        filename = "_".join(filename.split("_")[1:])
    output_filepath = os.path.join(output_dir, filename)
    new_df.to_csv(output_filepath, index=False)


def make_aecnn_table(cnn_attn_file):
    count_d4b = {
        "rule_0": 0,
        "rule_1": 0,
        "rule_2": 0,
        "rule_3": 0,
        "rule_4": 0,
    }
    df = pd.read_csv(cnn_attn_file)
    new_df = df.apply(
        pred_ss_based_on_bio, axis=1, args=(count_d4b,), result_type="expand"
    )
    column_names = [
        "time",
        "rule_0",
        "rule_1",
        "rule_2",
        "rule_3",
        "rule_4",
        "y_pred",
        "y_true",
        "up_flags",
        "truth_included",
    ]
    # new_df = new_df.rename(columns={0: "rule_a", 1: "rule_b"})
    new_df = new_df.rename(
        columns={key: val for key, val in enumerate(column_names)}
    )
    output_dir = os.path.join(os.environ["sleep"], "logs", "aecnn")
    if not os.path.exists(output_dir):
        print(f"{output_dir}を作成します")
        os.makedirs(output_dir)
    filename = os.path.split(cnn_attn_file)[1]
    # ファイル名が"output"から始まっている場合は取り除く
    if filename.startswith("output"):
        filename = "_".join(filename.split("_")[1:])
    output_filepath = os.path.join(output_dir, filename)
    new_df.to_csv(output_filepath, index=False)


def make_ccnn_table(cnn_attn_file, cnn_file):
    count_d4c = {
        "rule_0": 0,
        "rule_1": 0,
        "rule_2": 0,
        "rule_3": 0,
        "rule_4": 0,
    }
    # 同じ被験者データを読み込んでいることを確認
    try:
        assert os.path.split(cnn_attn_file)[1] == os.path.split(cnn_file)[1]
    except AssertionError as AE:
        print(f"assertion error: {AE}")
        sys.exit(1)

    attn_df = pd.read_csv(cnn_attn_file)
    no_attn_df = pd.read_csv(cnn_file)
    # 列名が被るのでプレフィクスを追加する
    attn_df = attn_df.add_prefix("attn_")
    no_attn_df = no_attn_df.add_prefix("no_attn_")
    # 結合
    new_df = pd.concat([attn_df, no_attn_df], axis=1)
    new_df = new_df.apply(
        pred_ss_custom, axis=1, args=(count_d4c,), result_type="expand"
    )
    column_names = [
        "time",
        "rule_0",
        "rule_1",
        "rule_2",
        "rule_3",
        "rule_4",
        "y_pred",
        "y_true",
        "up_flags",
        "truth_included",
    ]
    new_df = new_df.rename(
        columns={key: val for key, val in enumerate(column_names)}
    )
    output_dir = os.path.join(os.environ["sleep"], "logs", "ccnn")
    if not os.path.exists(output_dir):
        print(f"{output_dir}を作成します")
        os.makedirs(output_dir)
    filename = os.path.split(cnn_attn_file)[1]
    # ファイル名が"output"から始まっている場合は取り除く
    if filename.startswith("output"):
        filename = "_".join(filename.split("_")[1:])
    output_filepath = os.path.join(output_dir, filename)
    new_df.to_csv(output_filepath, index=False)


def make_eenn_table(enn_file):
    # カウンタのための辞書
    count_d4a = {"rule_0": 0, "rule_1": 0, "rule_2": 0}
    df = pd.read_csv(enn_file)
    new_df = df.apply(
        pred_ss_based_on_unc, args=(count_d4a,), axis=1, result_type="expand"
    )
    column_names = [
        "time",
        "rule_0",
        "rule_1",
        "rule_2",
        "y_pred",
        "y_true",
        "up_flags",
        "truth_included",
    ]
    new_df = new_df.rename(
        columns={key: val for key, val in enumerate(column_names)}
    )
    output_dir = os.path.join(os.environ["sleep"], "logs", "eenn")
    if not os.path.exists(output_dir):
        print(f"{output_dir}を作成します")
        os.makedirs(output_dir)
    filename = os.path.split(enn_file)[1]
    # ファイル名が"output"から始まっている場合は取り除く
    if filename.startswith("output"):
        filename = "_".join(filename.split("_")[1:])
    output_filepath = os.path.join(output_dir, filename)
    new_df.to_csv(output_filepath, index=False)


# これをしないと並列処理でランタイムエラーが出る
if __name__ == "__main__":
    # 提案手法1(ecnn)
    df4enn_b = None
    cnn_filelist = sorted(cnn_filelist)
    p = Pool(cpu_count())
    for cnn_file in tqdm(cnn_filelist):
        p.apply_async(make_ecnn_table, args=(cnn_file,))
    p.close()
    p.join()

    # 提案手法2(aecnn)
    df4enn_b = None
    cnn_attn_filelist = sorted(cnn_attn_filelist)
    p = Pool(cpu_count())
    for cnn_attn_file in tqdm(cnn_attn_filelist):
        p.apply_async(make_aecnn_table, args=(cnn_attn_file,))
    p.close()
    p.join()

    # 提案手法3(ccnn)
    df4enn_c = None
    cnn_filelist = sorted(cnn_filelist)
    p = Pool(cpu_count())
    for cnn_attn_file, cnn_file in tqdm(zip(cnn_attn_filelist, cnn_filelist)):
        p.apply_async(make_ccnn_table, args=(cnn_attn_file, cnn_file))
    p.close()
    p.join()

    # 提案手法4(eenn)
    df4enn_a = None
    enn_filelist = sorted(enn_filelist)
    p = Pool(cpu_count())
    for enn_file in tqdm(enn_filelist):
        p.apply_async(make_eenn_table, args=(enn_file,))
    p.close()
    p.join()
