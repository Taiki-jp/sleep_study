# 2クラス分類の結果をマージする
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 5段階(2段ENN以外)の結果がまとめてありcsv
testname_and_5stage_path = os.path.join(
    os.environ["git"], "sleep_study", "gallery", "testname_input.csv"
)
main_df = pd.read_csv(testname_and_5stage_path)

# 2クラス分類の統合後のパス
bin_dir = os.path.join(os.environ["sleep"], "logs", "bin_summary", "*.csv")
# 提案手法全てのメトリクスを出したパス
# bin_dir = os.path.join(
#     os.environ["sleep"], "logs", "ouput_metrics_from_ss", "*.csv"
# )
bin_filelist = glob(bin_dir)
enn_acc_list = list()
p1_acc_f_list = [[], [], [], [], [], []]
p1_acc_list = list()
p1_nr1_f_list = list()
p1_nr2_f_list = list()
p1_nr34_f_list = list()
p1_rem_f_list = list()
p1_wake_f_list = list()
p2_acc_f_list = [[], [], [], [], [], []]
p2_acc_list = list()
p2_nr1_f_list = list()
p2_nr2_f_list = list()
p2_nr34_f_list = list()
p2_rem_f_list = list()
p2_wake_f_list = list()
p3_acc_f_list = [[], [], [], [], [], []]
p3_acc_list = list()
p3_nr1_f_list = list()
p3_nr2_f_list = list()
p3_nr34_f_list = list()
p3_rem_f_list = list()
p3_wake_f_list = list()
p4_acc_f_list = [[], [], [], [], [], []]
p4_nr1_f_list = list()
p4_nr2_f_list = list()
p4_nr34_f_list = list()
p4_rem_f_list = list()
p4_wake_f_list = list()
p5_acc_f_list = [[], [], [], [], [], []]
p6_acc_f_list = [[], [], [], [], [], []]
p7_acc_f_list = [[], [], [], [], [], []]
p8_acc_f_list = [[], [], [], [], [], []]

# 2段ENNの一致率の結果のパス
nidan_dir = os.path.join(os.environ["sleep"], "tmps")

for bin_file in bin_filelist:
    bin_df = pd.read_csv(bin_file)
    # 0, 4, 8, 12, 16番目を取り出す
    accuracy = bin_df.loc[:, "accuracy"]
    ss = bin_df.loc[:, ["nr34", "nr2", "nr1", "rem", "wake"]]
    # print(accuracy)
    my_append = lambda list_type, val: list_type.append(val)
    tmp = map(
        my_append,
        [
            p1_acc_f_list[0],
            p2_acc_f_list[0],
            p3_acc_f_list[0],
            p4_acc_f_list[0],
            p5_acc_f_list[0],
            p6_acc_f_list[0],
            p7_acc_f_list[0],
            p8_acc_f_list[0],
            p1_acc_f_list[1],
            p1_acc_f_list[2],
            p1_acc_f_list[3],
            p1_acc_f_list[4],
            p1_acc_f_list[5],
            p2_acc_f_list[1],
            p2_acc_f_list[2],
            p2_acc_f_list[3],
            p2_acc_f_list[4],
            p2_acc_f_list[5],
            p3_acc_f_list[1],
            p3_acc_f_list[2],
            p3_acc_f_list[3],
            p3_acc_f_list[4],
            p3_acc_f_list[5],
            p4_acc_f_list[1],
            p4_acc_f_list[2],
            p4_acc_f_list[3],
            p4_acc_f_list[4],
            p4_acc_f_list[5],
            p5_acc_f_list[1],
            p5_acc_f_list[2],
            p5_acc_f_list[3],
            p5_acc_f_list[4],
            p5_acc_f_list[5],
            p6_acc_f_list[1],
            p6_acc_f_list[2],
            p6_acc_f_list[3],
            p6_acc_f_list[4],
            p6_acc_f_list[5],
            p7_acc_f_list[1],
            p7_acc_f_list[2],
            p7_acc_f_list[3],
            p7_acc_f_list[4],
            p7_acc_f_list[5],
            p8_acc_f_list[1],
            p8_acc_f_list[2],
            p8_acc_f_list[3],
            p8_acc_f_list[4],
            p8_acc_f_list[5],
        ],
        [
            accuracy[0],
            accuracy[4],
            accuracy[8],
            accuracy[12],
            accuracy[16],
            accuracy[20],
            accuracy[24],
            accuracy[28],
            ss.iloc[2, 0],
            ss.iloc[2, 1],
            ss.iloc[2, 2],
            ss.iloc[2, 3],
            ss.iloc[2, 4],
            ss.iloc[6, 0],
            ss.iloc[6, 1],
            ss.iloc[6, 2],
            ss.iloc[6, 3],
            ss.iloc[6, 4],
            ss.iloc[10, 0],
            ss.iloc[10, 1],
            ss.iloc[10, 2],
            ss.iloc[10, 3],
            ss.iloc[10, 4],
            ss.iloc[14, 0],
            ss.iloc[14, 1],
            ss.iloc[14, 2],
            ss.iloc[14, 3],
            ss.iloc[14, 4],
            ss.iloc[18, 0],
            ss.iloc[18, 1],
            ss.iloc[18, 2],
            ss.iloc[18, 3],
            ss.iloc[18, 4],
            ss.iloc[22, 0],
            ss.iloc[22, 1],
            ss.iloc[22, 2],
            ss.iloc[22, 3],
            ss.iloc[22, 4],
            ss.iloc[26, 0],
            ss.iloc[26, 1],
            ss.iloc[26, 2],
            ss.iloc[26, 3],
            ss.iloc[26, 4],
            ss.iloc[30, 0],
            ss.iloc[30, 1],
            ss.iloc[30, 2],
            ss.iloc[30, 3],
            ss.iloc[30, 4],
        ],
    )
    # tmp
    list(tmp)

# acc_list4nidan_enn = list()
# acc_list4nidan_enn_base = list()
# for dir_name, _, filename in os.walk(nidan_dir):
#     if "acc_of_high_unc_datas.csv" in filename:
#         nidan_enn_filepath = os.path.join(
#             dir_name, "acc_of_high_unc_datas.csv"
#         )
#         nidan_df = pd.read_csv(nidan_enn_filepath)
#         acc_list4nidan_enn_base.append(nidan_df.iloc[0, 3])
#         acc_list4nidan_enn.append(nidan_df.iloc[0, 5])

# enn_df = pd.DataFrame(
#     {"enn_base": acc_list4nidan_enn_base, "enn_merged": acc_list4nidan_enn}
# )

bin_df = pd.DataFrame(
    {
        "cnn_attn_acc": p1_acc_f_list[0],
        "cnn_attn_f_nr34": p1_acc_f_list[1],
        "cnn_attn_f_nr2": p1_acc_f_list[2],
        "cnn_attn_f_nr1": p1_acc_f_list[3],
        "cnn_attn_f_rem": p1_acc_f_list[4],
        "cnn_attn_f_wake": p1_acc_f_list[5],
        "cnn_no_attn_acc": p2_acc_f_list[0],
        "cnn_no_attn_f_nr34": p2_acc_f_list[1],
        "cnn_no_attn_f_nr2": p2_acc_f_list[2],
        "cnn_no_attn_f_nr1": p2_acc_f_list[3],
        "cnn_no_attn_f_rem": p2_acc_f_list[4],
        "cnn_no_attn_f_wake": p2_acc_f_list[5],
        "enn_acc": p3_acc_f_list[0],
        "enn_f_nr34": p3_acc_f_list[1],
        "enn_f_nr2": p3_acc_f_list[2],
        "enn_f_nr1": p3_acc_f_list[3],
        "enn_f_rem": p3_acc_f_list[4],
        "enn_f_wake": p3_acc_f_list[5],
        "aecnn_acc": p4_acc_f_list[0],
        "aecnn_f_nr34": p4_acc_f_list[1],
        "aecnn_f_nr2": p4_acc_f_list[2],
        "aecnn_f_nr1": p4_acc_f_list[3],
        "aecnn_f_rem": p4_acc_f_list[4],
        "aecnn_f_wake": p4_acc_f_list[5],
        "ecnn_acc": p5_acc_f_list[0],
        "ecnn_f_nr34": p5_acc_f_list[1],
        "ecnn_f_nr2": p5_acc_f_list[2],
        "ecnn_f_nr1": p5_acc_f_list[3],
        "ecnn_f_rem": p5_acc_f_list[4],
        "ecnn_f_wake": p5_acc_f_list[5],
        "ccnn_acc": p6_acc_f_list[0],
        "ccnn_f_nr34": p6_acc_f_list[1],
        "ccnn_f_nr2": p6_acc_f_list[2],
        "ccnn_f_nr1": p6_acc_f_list[3],
        "ccnn_f_rem": p6_acc_f_list[4],
        "ccnn_f_wake": p6_acc_f_list[5],
        "eenn_acc": p7_acc_f_list[0],
        "eenn_f_nr34": p7_acc_f_list[1],
        "eenn_f_nr2": p7_acc_f_list[2],
        "eenn_f_nr1": p7_acc_f_list[3],
        "eenn_f_rem": p7_acc_f_list[4],
        "eenn_f_wake": p7_acc_f_list[5],
        "denn_acc": p8_acc_f_list[0],
        "denn_f_nr34": p8_acc_f_list[1],
        "denn_f_nr2": p8_acc_f_list[2],
        "denn_f_nr1": p8_acc_f_list[3],
        "denn_f_rem": p8_acc_f_list[4],
        "denn_f_wake": p8_acc_f_list[5],
    }
)

output_df = pd.concat(
    [main_df, bin_df],
    axis=1,
)
output_df.to_csv("tmp.csv", index=False)
