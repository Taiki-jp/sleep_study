# 2クラス分類の結果をマージする
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ennのパス
enn_filedir = os.path.join(
    os.environ["git"], "sleep_study", "bin_enn", "*.csv"
)
# cnn+attnのパス
cnn_attn_filedir = os.path.join(
    os.environ["git"], "sleep_study", "cnn_attn", "*.csv"
)
# cnnのパス
cnn_filedir = os.path.join(
    os.environ["git"], "sleep_study", "cnn_no_attn", "*.csv"
)

enn_filelist = glob(enn_filedir)
cnn_attn_filelist = glob(cnn_attn_filedir)
cnn_filelist = glob(cnn_filedir)


def merge_rule_a(series: pd.Series):
    bin_pred = series[
        ["nr34_pred", "nr2_pred", "nr1_pred", "rem_pred", "wake_pred"]
    ]
    bin_pred = bin_pred.to_numpy()

    # 一つ目のルール
    if sum(bin_pred) == 1:
        return np.argmax(bin_pred)

    # 二つ目のルール
    elif sum(bin_pred) > 1:
        flag_unc = series[
            ["nr34_unc", "nr2_unc", "nr1_unc", "rem_unc", "wake_unc"]
        ]
        # フラグが立っている場所だけ残す
        flag_unc = flag_unc.to_numpy() * bin_pred
        # 不確実性が最小の睡眠段階を抽出する
        #             print(flag_unc == np.min(flag_unc[np.nonzero(flag)]))
        #             print(np.argmax(flag_unc == np.min(flag_unc[np.nonzero(flag)])))
        return np.argmax(flag_unc == np.min(flag_unc[np.nonzero(bin_pred)]))

    # 三つ目のルール
    else:
        assert sum(bin_pred) == 0
        # NR2を返す
        return 1


def merge_rule_b(series: pd.Series):
    bin_pred = series[
        ["nr34_pred", "nr2_pred", "nr1_pred", "rem_pred", "wake_pred"]
    ]
    bin_pred = bin_pred.to_numpy()
    # 0. ひとつだけPositiveならそのクラス
    if sum(bin_pred) == 1:
        return np.argmax(bin_pred)

    # 1. NR2, remがPositiveならrem
    elif bin_pred[2] == 1 and bin_pred[4] == 1:
        return 3

    # 2. nr1, wakeがpositiveならwake
    elif bin_pred[2] == 1 and bin_pred[4] == 1:
        return 4

    # 3. positiveのなかで優先度の高いもので並び替え
    elif int(sum(bin_pred)) > 1:
        if bin_pred[1] == 1:
            return 1
        elif bin_pred[3] == 1:
            return 3
        elif bin_pred[2] == 1:
            return 2
        elif bin_pred[4] == 1:
            return 4
        elif bin_pred[0] == 1:
            return 0
        else:
            raise AssertionError
    else:
        return 1


def merge_rule_c(attn_df: pd.DataFrame, no_attn_df: pd.DataFrame):
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
                _bin_pred_attn[0],
                _bin_pred_no_attn[1],
                _bin_pred_no_attn[2],
                _bin_pred_attn[3],
                _bin_pred_no_attn[4],
            ]
        )
        if sum(array4case_1) == 1:
            y_pred.append(np.argmax(_bin_pred_attn))
        elif (
            _bin_pred_no_attn[0] == 1
            or _bin_pred_no_attn[1] == 1
            or _bin_pred_no_attn[3] == 1
        ):
            y_pred.append(np.argmax(array4case_1))

        elif (
            _bin_pred_attn[2] == 1
            or _bin_pred_attn[4] == 1
            and pd.concat([_bin_pred_no_attn, _bin_pred_attn], axis=0).sum()
            == 2
        ):
            y_pred.append(4)
        elif (
            pd.concat([_bin_pred_no_attn, _bin_pred_attn], axis=0).sum() > 1
            and sum(array4case_1) > 1
        ):
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
            y_pred.append(0)
    return pd.DataFrame(y_pred, columns=["y_pred"])


def merge_rule_d(attn_df: pd.DataFrame, no_attn_df: pd.DataFrame):
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
                _bin_pred_attn[0],
                _bin_pred_no_attn[1],
                _bin_pred_no_attn[2],
                _bin_pred_attn[3],
                _bin_pred_no_attn[4],
            ]
        )
        if sum(array4case_1) == 1:
            y_pred.append(np.argmax(_bin_pred_attn))
        elif (
            _bin_pred_attn[0] == 1
            or _bin_pred_no_attn[1] == 1
            or _bin_pred_attn[3] == 1
        ):
            y_pred.append(np.argmax(array4case_1))

        elif (
            _bin_pred_no_attn[2] == 1
            or _bin_pred_attn[4] == 1
            and pd.concat([_bin_pred_no_attn, _bin_pred_attn], axis=0).sum()
            == 2
        ):
            y_pred.append(4)
        elif (
            pd.concat([_bin_pred_no_attn, _bin_pred_attn], axis=0).sum() > 1
            and sum(array4case_1) > 1
        ):
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
            y_pred.append(0)
    return pd.DataFrame(y_pred, columns=["y_pred"])


# 提案手法4
for enn_file in enn_filelist:
    print(f"load {enn_file}")
    df = pd.read_csv(enn_file)
    y_pred_merged_by_a = df.apply(merge_rule_a, axis=1)
    y_pred_merged_by_b = df.apply(merge_rule_b, axis=1)
    new_df = pd.concat([df, y_pred_merged_by_a, y_pred_merged_by_b], axis=1)
    new_df = new_df.rename(columns={0: "rule_a", 1: "rule_b"})
    new_df = new_df.drop(columns="Unnamed: 0")
    output_dir = os.path.join(os.environ["sleep"], "logs", "bin_enn_output")
    if not os.path.exists(output_dir):
        print(f"{output_dir}を作成します")
        os.makedirs(output_dir)
    filename = os.path.split(enn_file)[1]
    output_filepath = os.path.join(output_dir, filename)
    new_df.to_csv(output_filepath, index=False)

# 提案手法1
for cnn_file in cnn_filelist:
    print(f"load {cnn_file}")
    df = pd.read_csv(cnn_file)
    # y_pred_merged_by_a = df.apply(merge_rule_a, axis=1)
    y_pred_merged_by_b = df.apply(merge_rule_b, axis=1)
    # new_df = pd.concat([df, y_pred_merged_by_a, y_pred_merged_by_b], axis=1)
    new_df = pd.concat([df, y_pred_merged_by_b], axis=1)
    # new_df = new_df.rename(columns={0: "rule_a", 1: "rule_b"})
    new_df = new_df.rename(columns={0: "rule_b"})
    new_df = new_df.drop(columns="Unnamed: 0")
    output_dir = os.path.join(os.environ["sleep"], "logs", "proposed_01")
    if not os.path.exists(output_dir):
        print(f"{output_dir}を作成します")
        os.makedirs(output_dir)
    filename = os.path.split(cnn_file)[1]
    output_filepath = os.path.join(output_dir, filename)
    new_df.to_csv(output_filepath, index=False)

# 提案手法2
for cnn_attn_file in cnn_attn_filelist:
    print(f"load {cnn_attn_file}")
    df = pd.read_csv(cnn_attn_file)
    # y_pred_merged_by_a = df.apply(merge_rule_a, axis=1)
    y_pred_merged_by_b = df.apply(merge_rule_b, axis=1)
    # new_df = pd.concat([df, y_pred_merged_by_a, y_pred_merged_by_b], axis=1)
    new_df = pd.concat([df, y_pred_merged_by_b], axis=1)
    # new_df = new_df.rename(columns={0: "rule_a", 1: "rule_b"})
    new_df = new_df.rename(columns={0: "rule_b"})
    new_df = new_df.drop(columns="Unnamed: 0")
    output_dir = os.path.join(os.environ["sleep"], "logs", "proposed_02")
    if not os.path.exists(output_dir):
        print(f"{output_dir}を作成します")
        os.makedirs(output_dir)
    filename = os.path.split(cnn_attn_file)[1]
    output_filepath = os.path.join(output_dir, filename)
    new_df.to_csv(output_filepath, index=False)

# 旧提案手法3
for cnn_attn_file, cnn_file in zip(cnn_attn_filelist, cnn_filelist):
    print(f"load {cnn_attn_file}")
    print(f"load {cnn_file}")
    # 同じ被験者データを読み込んでいることを確認
    try:
        assert os.path.split(cnn_attn_file)[1] == os.path.split(cnn_file)[1]
    except AssertionError as AE:
        print(f"assertion error: {AE}")
        sys.exit(1)

    attn_df = pd.read_csv(cnn_attn_file)
    no_attn_df = pd.read_csv(cnn_file)
    new_df = merge_rule_c(attn_df=attn_df, no_attn_df=no_attn_df)
    # y_pred_merged_by_a = df.apply(merge_rule_a, axis=1)
    new_df = pd.concat([attn_df, no_attn_df, new_df], axis=1)
    new_df = new_df.rename(columns={0: "rule_c"})
    new_df = new_df.drop(columns="Unnamed: 0")
    output_dir = os.path.join(os.environ["sleep"], "logs", "proposed_03")
    if not os.path.exists(output_dir):
        print(f"{output_dir}を作成します")
        os.makedirs(output_dir)
    filename = os.path.split(cnn_attn_file)[1]
    output_filepath = os.path.join(output_dir, filename)
    new_df.to_csv(output_filepath, index=False)

# 提案手法3
for cnn_attn_file, cnn_file in zip(cnn_attn_filelist, cnn_filelist):
    print(f"load {cnn_attn_file}")
    print(f"load {cnn_file}")
    # 同じ被験者データを読み込んでいることを確認
    try:
        assert os.path.split(cnn_attn_file)[1] == os.path.split(cnn_file)[1]
    except AssertionError as AE:
        print(f"assertion error: {AE}")
        sys.exit(1)

    attn_df = pd.read_csv(cnn_attn_file)
    no_attn_df = pd.read_csv(cnn_file)
    new_df = merge_rule_d(attn_df=attn_df, no_attn_df=no_attn_df)
    # y_pred_merged_by_a = df.apply(merge_rule_a, axis=1)
    new_df = pd.concat([attn_df, no_attn_df, new_df], axis=1)
    new_df = new_df.rename(columns={0: "rule_d"})
    new_df = new_df.drop(columns="Unnamed: 0")
    output_dir = os.path.join(os.environ["sleep"], "logs", "proposed_04")
    if not os.path.exists(output_dir):
        print(f"{output_dir}を作成します")
        os.makedirs(output_dir)
    filename = os.path.split(cnn_attn_file)[1]
    output_filepath = os.path.join(output_dir, filename)
    new_df.to_csv(output_filepath, index=False)
