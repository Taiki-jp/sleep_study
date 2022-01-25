# 2クラス分類の結果をマージする
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cv2 import sort
from sklearn.metrics import classification_report, confusion_matrix

# 提案手法1のパス
proposed01_pathlist = os.path.join(
    os.environ["sleep"], "logs", "proposed_01", "*.csv"
)
# 提案手法2のパス
proposed02_pathlist = os.path.join(
    os.environ["sleep"], "logs", "proposed_02", "*.csv"
)
# 旧提案手法3のパス
proposed03_pathlist = os.path.join(
    os.environ["sleep"], "logs", "proposed_03", "*.csv"
)
# 新提案手法3のパス
proposed04_pathlist = os.path.join(
    os.environ["sleep"], "logs", "proposed_04", "*.csv"
)
# ennのパス
enn_pathlist = os.path.join(
    os.environ["sleep"], "logs", "bin_enn_output", "*.csv"
)

proposed01_filelist = glob(proposed01_pathlist)
proposed02_filelist = glob(proposed02_pathlist)
proposed03_filelist = glob(proposed03_pathlist)
proposed04_filelist = glob(proposed04_pathlist)
enn_filelist = glob(enn_pathlist)
# ennの場合はoutput_*始まりのときに先頭6文字を無視する <= いらない（prefixがついていてもソートは出来るから）
# if os.path.split(enn_filelist[0])[1].startswith("output_"):
#     enn_filelist = [
#         os.path.join(
#             os.path.split(trimmed_enn_filename)[0],
#             os.path.split(trimmed_enn_filename)[1][len("output_") :],
#         )
#         for trimmed_enn_filename in enn_filelist
#     ]
# 昇順にソートする
proposed01_filelist = sorted(proposed01_filelist)
proposed02_filelist = sorted(proposed02_filelist)
proposed03_filelist = sorted(proposed03_filelist)
proposed04_filelist = sorted(proposed04_filelist)
enn_filelist = sorted(enn_filelist)
for pp01, pp02, pp03, pp04, enn in zip(
    proposed01_filelist,
    proposed02_filelist,
    proposed03_filelist,
    proposed04_filelist,
    enn_filelist,
):
    # ファイル名が全てのファイルリストで同じ順でループが回されていることの確認
    try:
        assert (
            os.path.split(pp01)[1]
            == os.path.split(pp02)[1]
            == os.path.split(pp03)[1]
            == os.path.split(pp04)[1]
        )
    except AssertionError as AE:
        raise AE
    pp01_df, pp02_df, pp03_df, pp04_df, enn_df = map(
        lambda filename: pd.read_csv(filename), [pp01, pp02, pp03, pp04, enn]
    )
    # print(pp01_df.head())
    # print(pp03_df.head())
    time_series = pp01_df.loc[:, "time"]
    y_tue_series = pp01_df.loc[:, "y_true"]
    pp01_pred = pp01_df.loc[:, "rule_b"]
    pp02_pred = pp02_df.loc[:, "rule_b"]
    pp03_pred = pp03_df.loc[:, "y_pred"]
    pp04_pred = pp04_df.loc[:, "y_pred"]
    enn_pred = enn_df.loc[:, "rule_a"]
    # print(y_tue_series)
    # print(len(pp03_pred))
    # 混同行列の作成
    # pp01_cm = confusion_matrix(y_true=y_tue_series, y_pred=pp01_pred)
    # print(pp01_cm)
    # 試す
    try:
        pp01_report = classification_report(
            y_true=y_tue_series,
            y_pred=pp01_pred,
            target_names=["nr34", "nr2", "nr1", "rem", "wake"],
            output_dict=True,
        )
    except:
        pp01_report = classification_report(
            y_true=y_tue_series,
            y_pred=pp01_pred,
            target_names=["nr2", "nr1", "rem", "wake"],
            output_dict=True,
        )
    try:
        pp02_report = classification_report(
            y_true=y_tue_series,
            y_pred=pp02_pred,
            target_names=["nr34", "nr2", "nr1", "rem", "wake"],
            output_dict=True,
        )
    except:
        pp02_report = classification_report(
            y_true=y_tue_series,
            y_pred=pp02_pred,
            target_names=["nr2", "nr1", "rem", "wake"],
            output_dict=True,
        )
    try:
        pp03_report = classification_report(
            y_true=y_tue_series,
            y_pred=pp03_pred,
            target_names=["nr34", "nr2", "nr1", "rem", "wake"],
            output_dict=True,
        )
    except:
        pp03_report = classification_report(
            y_true=y_tue_series,
            y_pred=pp03_pred,
            target_names=["nr2", "nr1", "rem", "wake"],
            output_dict=True,
        )
    try:
        pp04_report = classification_report(
            y_true=y_tue_series,
            y_pred=pp04_pred,
            target_names=["nr34", "nr2", "nr1", "rem", "wake"],
            output_dict=True,
        )
    except:
        pp04_report = classification_report(
            y_true=y_tue_series,
            y_pred=pp04_pred,
            target_names=["nr2", "nr1", "rem", "wake"],
            output_dict=True,
        )
    try:
        enn_report = classification_report(
            y_true=y_tue_series,
            y_pred=enn_pred,
            target_names=["nr34", "nr2", "nr1", "rem", "wake"],
            output_dict=True,
        )
    except:
        enn_report = classification_report(
            y_true=y_tue_series,
            y_pred=enn_pred,
            target_names=["nr2", "nr1", "rem", "wake"],
            output_dict=True,
        )
    pp01_df = pd.DataFrame(pp01_report)
    # print(pp01_df.head())
    metrics = ["precision", "recall", "f1-score", "support"]
    pp01_df = pp01_df.rename(
        {__metrics: __metrics + "_pp01" for __metrics in metrics}
    )
    # pp01_df = pp01_df.T
    # pp01_df = pp01_df.add_prefix("pp01_")
    pp02_df = pd.DataFrame(pp02_report)
    pp02_df = pp02_df.rename(
        {__metrics: __metrics + "_pp02" for __metrics in metrics}
    )
    # pp02_df = pp02_df.T
    # pp02_df = pp02_df.add_prefix("pp02_")
    pp03_df = pd.DataFrame(pp03_report)
    pp03_df = pp03_df.rename(
        {__metrics: __metrics + "_pp03" for __metrics in metrics}
    )
    # pp03_df = pp03_df.T
    # pp03_df = pp03_df.add_prefix("pp03_")
    pp04_df = pd.DataFrame(pp04_report)
    pp04_df = pp04_df.rename(
        {__metrics: __metrics + "_pp04" for __metrics in metrics}
    )
    enn_df = pd.DataFrame(enn_report)
    enn_df = enn_df.rename(
        {__metrics: __metrics + "_enn" for __metrics in metrics}
    )
    output_df = pd.concat([pp01_df, pp02_df, pp03_df, pp04_df, enn_df], axis=0)
    output_dir = os.path.join(os.environ["sleep"], "logs", "bin_summary")
    if not os.path.exists(output_dir):
        print(f"{output_dir}を作成します")
        os.makedirs(output_dir)
    filename = os.path.split(pp01)[1]
    output_filepath = os.path.join(output_dir, filename)
    output_df.to_csv(output_filepath)
    # break
