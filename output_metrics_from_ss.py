# 2クラス分類の結果をマージする
import os
import sys
from glob import glob
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cv2 import sort
from sklearn.metrics import classification_report, confusion_matrix

# CNN(attention)
cnn_attn_pathlist = os.path.join(
    os.environ["git"], "sleep_study", "cnn_attn_5stage", "*", "ss_5class.csv"
)
# CNN(no-attention)
cnn_no_attn_pathlist = os.path.join(
    os.environ["git"],
    "sleep_study",
    "cnn_noattn_5stage",
    "*",
    "ss_5class.csv",
)
# ENN
enn_pathlist = os.path.join(
    os.environ["git"], "sleep_study", "enn_all_result", "*", "ss_and_unc.csv"
)
# AECNN
aecnn_pathlist = os.path.join(
    os.environ["sleep"], "logs", "proposed01", "*.csv"
)
# ECNN
ecnn_pathlist = os.path.join(
    os.environ["sleep"], "logs", "proposed02", "*.csv"
)
# CCNN
ccnn_pathlist = os.path.join(
    os.environ["sleep"], "logs", "proposed03", "*.csv"
)
# EENN
eenn_pathlist = os.path.join(
    os.environ["sleep"], "logs", "bin_enn_output", "*.csv"
)
# DENN
denn_pathlist = os.path.join(
    os.environ["git"],
    "sleep_study",
    "enn_all_result",
    "*",
    "merged_result.csv",
)
pathlists = [
    cnn_attn_pathlist,
    cnn_no_attn_pathlist,
    enn_pathlist,
    aecnn_pathlist,
    ecnn_pathlist,
    ccnn_pathlist,
    eenn_pathlist,
    denn_pathlist,
]

filelists = map(sorted, list(map(glob, pathlists)))
filelists = list(filelists)


def calc_metrics(
    df: pd.DataFrame, label: str
) -> Tuple[float, float, float, float]:
    y_true = df["y_true"]
    if label == "enn":
        y_pred = df["base_pred"]
    elif label == "aecnn" or label == "ecnn":
        y_pred = df["rule_b"]
    elif label == "eenn":
        y_pred = df["rule_a"]
    else:
        y_pred = df["y_pred"]
    try:
        report = classification_report(
            y_true=y_true,
            y_pred=y_pred,
            target_names=["nr34", "nr2", "nr1", "rem", "wake"],
            output_dict=True,
        )
    except:
        report = classification_report(
            y_true=y_true,
            y_pred=y_pred,
            target_names=["nr2", "nr1", "rem", "wake"],
            output_dict=True,
        )
    return pd.DataFrame(report)


for cnn_attn, cnn_no_attn, enn, aecnn, ecnn, ccnn, eenn, denn in zip(
    filelists[0],
    filelists[1],
    filelists[2],
    filelists[3],
    filelists[4],
    filelists[5],
    filelists[6],
    filelists[7],
):
    _filelists = (cnn_attn, cnn_no_attn, enn, aecnn, ecnn, ccnn, eenn, denn)
    df_list = list(map(pd.read_csv, _filelists))
    df_list_label = [
        "cnn_attn",
        "cnn_no_attn",
        "enn",
        "aecnn",
        "ecnn",
        "ccnn",
        "eenn",
        "denn",
    ]
    metrics = ["precision", "recall", "f1-score", "support"]
    df_d: Dict[str, pd.DataFrame] = {
        _key: _val for _key, _val in zip(df_list_label, df_list)
    }
    report_d: Dict[str, pd.DataFrame] = {_key: "" for _key in df_list_label}
    for label, df in df_d.items():
        report_d[label] = calc_metrics(df, label)
        report_d[label] = report_d[label].rename(
            {__metrics: __metrics + f"_{label}" for __metrics in metrics}
        )

    output_df = pd.concat([__df for __df in report_d.values()], axis=0)
    output_dir = os.path.join(
        os.environ["sleep"], "logs", "ouput_metrics_from_ss"
    )
    # aecnnから被験者名を抽出
    filename = os.path.split(aecnn)[1]
    output_path = os.path.join(output_dir, filename)
    if not os.path.exists(output_dir):
        print("make dirs")
        os.makedirs(output_dir)
    output_df.to_csv(output_path)
    # break
