# 2クラス分類の結果をマージする
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 不確実性の出力が含まれるファイルパス（2クラス分類）
# unc_dir = os.path.join(os.environ["git"], "sleep_study", "bin_enn", "*.csv")
# unc_files = glob(unc_dir)
# unc_files = sorted(unc_files)
# ss = ["nr34", "nr2", "nr1", "rem", "wake"]
# ss_added_suffix = [_ss + "_unc" for _ss in ss]
# concated_df = None

# for filepath in unc_files:
#     df = pd.read_csv(filepath)
#     df_mean = df.mean()
#     df_mean_ss = df_mean[ss_added_suffix]
#     if concated_df is not None:
#         concated_df = pd.concat([concated_df, df_mean_ss], axis=1)
#     else:
#         concated_df = df_mean_ss

# concated_df.to_csv("unc_average.csv")

# 不確実性の出力が含まれるファイルパス(5クラス分類)
unc_dir = os.path.join(os.environ["sleep"], "tmps")
concated_df = None
for dir_name, _, filename in os.walk(unc_dir):
    if "ss_and_unc.csv" in filename:
        nidan_enn_filepath = os.path.join(dir_name, "ss_and_unc.csv")
        nidan_df = pd.read_csv(nidan_enn_filepath)
        nidan_df_mean = nidan_df.mean()[["unc_base", "unc_pos"]]
        if concated_df is not None:
            concated_df = pd.concat([concated_df, nidan_df_mean], axis=1)
        else:
            concated_df = nidan_df_mean

concated_df.to_csv("unc_average_5stage.csv")
