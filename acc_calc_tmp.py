# 2クラス分類の結果をマージする
from data_analysis.utils import Mine


def main():
    return


if __name__ == "__main__":
    VERBOSE = 1
    # 提案手法（キー）とそのパスのための情報（バリュー）の辞書
    path_arg_d = {
        "cnn_attn": ["cnn_attn_5stage", "*.csv"],
        "cnn_no_attn": ["cnn_noattn_5stage", "*.csv"],
        "aecnn": ["aecnn", "*.csv"],
        "ecnn": ["ecnn", "*.csv"],
        "ccnn_ver1": ["ccnn", "*.csv"],
        "eenn": ["eenn", "*.csv"],
        "denn": ["denn", "*.csv"],
    }
    mine = Mine(path_arg_d, VERBOSE)
    mine.exec()
