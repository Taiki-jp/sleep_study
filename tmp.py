# 2クラス分類の結果をマージする
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# 統合ルールA
# 0. フラグが一本立っている場合はそれ
# 1. フラグが複数立っている場合一番不確実性の一番低いものを採用する
# 2. フラグが何も立っていない場合はNR2と判断する
def merge_rule_a(df):
    y_pred = list()
    used_list = list()
    table = df.to_numpy()
    for row in table:
        # flagの立っている一の情報を保持
        flag = row[1:6]
        # rule0
        if int(sum(flag)) == 1:
            y_pred.append(np.argmax(flag))
            used_list.append(0)
            continue
        # rule1
        elif int(sum(flag)) > 1:
            flag_unc = row[6:11]
            # フラグが立っている場所だけ残す
            flag_unc = flag_unc * flag
            # 不確実性が最小の睡眠段階を抽出する
            #             print(flag_unc == np.min(flag_unc[np.nonzero(flag)]))
            #             print(np.argmax(flag_unc == np.min(flag_unc[np.nonzero(flag)])))
            y_pred.append(
                np.argmax(flag_unc == np.min(flag_unc[np.nonzero(flag)]))
            )
            used_list.append(1)
            continue
        else:
            assert int(sum(flag)) == 0
            # NR2を返す
            y_pred.append(1)
            used_list.append(2)
    return y_pred, used_list


# 統合ルールB
# 0. ひとつだけPositiveならそのクラス
# 1. NR2, remがPositiveならrem
# 2. nr1, wakeがpositiveならwake
# 3. positiveのなかで優先度の高いもので並び替え
# 4. nr2
def merge_rule_b(df):
    y_pred = list()
    used_list = list()
    table = df.to_numpy()
    for row in table:
        # 0. ひとつだけPositiveならそのクラス
        flag = row[1:6]
        if int(sum(flag)) == 1:
            y_pred.append(np.argmax(flag))
            used_list.append(0)

        # 1. NR2, remがPositiveならrem
        elif row[2] == 1 and row[4] == 1:
            y_pred.append(3)
            used_list.append(1)

        # 2. nr1, wakeがpositiveならwake
        elif row[3] == 1 and row[5] == 1:
            y_pred.append(4)
            used_list.append(2)

        # 3. positiveのなかで優先度の高いもので並び替え
        elif int(sum(flag)) > 1:
            flag_unc = row[6:11]
            # フラグが立っている場所だけ残す
            flag_unc = flag_unc * flag
            # 不確実性が最小の睡眠段階を抽出する
            #             print(flag_unc == np.min(flag_unc[np.nonzero(flag)]))
            #             print(np.argmax(flag_unc == np.min(flag_unc[np.nonzero(flag)])))
            y_pred.append(
                np.argmax(flag_unc == np.min(flag_unc[np.nonzero(flag)]))
            )
            used_list.append(3)

        # 4. nr2
        else:
            assert int(sum(flag)) == 0
            # NR2を返す
            y_pred.append(1)
            used_list.append(4)
    return y_pred, used_list


def calc_acc(df):
    def __judge_acc_from_column_id(l):
        return sum(df[str(l[0])] == df[str(l[1])])

    def __calc_rule_correctness():
        # 11列目：正解データ，12列目：ruleAの予測，13列目：ruleBの予測
        # 14列目：使われたruleAのインデックス，15列目：使われたruleBのインデックス
        np_table = df.loc[:, ["11", "12", "13", "14", "15"]].to_numpy()
        # カウンタ用の配列を取得(2, 5)
        # (0, 5)->ruleAの正解カウンタ
        rule_correct_counter = np.zeros(10).reshape(2, 5)
        rule_wrong_counter = np.zeros(10).reshape(2, 5)
        for row in np_table:
            if row[0] == row[1]:
                rule_correct_counter[0, int(row[3])] += 1
            else:
                rule_wrong_counter[0, int(row[3])] += 1
            if row[0] == row[2]:
                rule_correct_counter[1, int(row[4])] += 1
            else:
                rule_wrong_counter[1, int(row[4])] += 1
        return rule_correct_counter.tolist(), rule_wrong_counter.tolist()

    acc_list = list(map(__judge_acc_from_column_id, [[11, 12], [11, 13]]))
    print("統合ルールA, Bの正解数", acc_list)
    row, _ = df.shape
    rcc, rwc = __calc_rule_correctness()
    return [_acc_list / row for _acc_list in acc_list], rcc, rwc


dir_2 = ("1_1", "1_2")
for __dir_2 in dir_2:
    filedir = os.path.join(
        os.environ["sleep"], "log", "bin_merge", __dir_2, "*.csv"
    )
    filelist = glob(filedir)
    for filepath in filelist:
        _, filename = os.path.split(filepath)
        df = pd.read_csv(filepath)
        y_pred_a, used_list_a = merge_rule_a(df)
        y_pred_b, used_list_b = merge_rule_b(df)
        y_pred = np.array([y_pred_a, y_pred_b]).T  # reshape(-1, 2)
        new_df = pd.DataFrame(
            np.hstack(
                [
                    df.to_numpy(),
                    y_pred,
                    np.array(used_list_a).reshape(-1, 1),
                    np.array(used_list_b).reshape(-1, 1),
                ]
            )
        )
        output_path = os.path.join(
            os.environ["sleep"], "log", f"bin_merge_{__dir_2}"
        )
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        new_df.to_csv(os.path.join(output_path, filename))

target_dir = ("1_1", "1_2")
acc_list = list()
rcc_list = list()
rwc_list = list()

for __target_dir in target_dir:
    acc_list_sub = list()
    rcc_list_sub = list()
    rwc_list_sub = list()
    filelist = os.path.join(
        os.environ["sleep"], "log", f"bin_merge_{__target_dir}", "*.csv"
    )
    filelist = glob(filelist)

    for filepath in filelist:
        df = pd.read_csv(filepath)
        acc_list, rcc, rwc = calc_acc(df)
        acc_list_sub.append(acc_list)
        rcc_list_sub.append(rcc)
        rwc_list_sub.append(rwc)
    acc_list.append(acc_list_sub)
    rcc_list.append(rcc_list_sub)
    rwc_list.append(rwc_list_sub)


# rcc_listの構造
# 0行目：r1:1の結果（全ての被験者）
# 0行0列目：r1:1の結果（一人の被験者）
rcc_array = np.array(rcc_list[0])
print("rcc_array.shape", rcc_array.shape)
rwc_array = np.array(rwc_list[0])
# すべての被験者で平均を計算
print(rcc_array.mean(axis=0))
print(rwc_array.mean(axis=0))
