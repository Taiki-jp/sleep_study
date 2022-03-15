# 2クラス分類の結果をマージする
import os
import sys
from glob import glob
from tabnanny import verbose

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from data_analysis.utils import Mine


def main():
    return


if __name__ == "__main__":
    VERBOSE = 1
    # 提案手法（キー）とそのパスのための情報（バリュー）の辞書
    path_arg_d = {
        "aecnn": ["aecnn", "*.csv"],
        "ecnn": ["ecnn", "*.csv"],
        "ccnn_ver1": ["ccnn", "*.csv"],
        # "ccnn_ver2": ["proposed_04", "*.csv"],
        "eenn": ["eenn", "*.csv"],
    }
    mine = Mine(path_arg_d, VERBOSE)
    mine.exec()
