# 2段ENNの結果をマージする
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 各被験者データが入っている親ディレクトリ
nidan_dir = os.path.join(os.environ["git"], "sleep_study", "nidan_enn")

acc_list4nidan_enn = list()
acc_list4nidan_enn_base = list()
for dir_name, _, filename in os.walk(nidan_dir):
    if "acc_of_high_unc_datas.csv" in filename:
        nidan_enn_filepath = os.path.join(
            dir_name, "acc_of_high_unc_datas.csv"
        )
        nidan_df = pd.read_csv(nidan_enn_filepath)
        acc_list4nidan_enn_base.append(nidan_df.iloc[0, 1])
        acc_list4nidan_enn.append(nidan_df.iloc[0, 2])

enn_df = pd.DataFrame(
    {"enn_base": acc_list4nidan_enn_base, "enn_sub": acc_list4nidan_enn}
)

enn_df.to_csv("tmp_enn_high_unc_acc.csv", index=False)
