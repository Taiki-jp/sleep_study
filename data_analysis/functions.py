import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from sklearn import datasets
from sklearn.decomposition import PCA
import sys
from sklearn.manifold import TSNE
from keras import backend as K


# ref : https://www.kaggle.com/kabirnagpal/xception-resnet-learn-how-to-stack
def recall_m(y_true, y_pred):

    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), 0.5), K.floatx())
    true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.clip(y_true, 0, 1))
    recall_ratio = true_positives / (possible_positives + K.epsilon())
    return recall_ratio


def precision_m(y_true, y_pred):

    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), 0.5), K.floatx())
    true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(y_pred)
    precision_ratio = true_positives / (predicted_positives + K.epsilon())
    return precision_ratio


def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# ref : http://inaz2.hatenablog.com/entry/2017/01/24/211331
# pca
def make_pca(
    data=None,
    pse_data=True,
    path=None,
    show_data=True,
    data_type=None,
    test_id=0,
):
    if pse_data:
        digits = datasets.load_digits()
        X_reduced = PCA(n_components=2).fit_transform(digits.data)
    else:
        if data_type == None:
            print("data_typeを指定してください")
            sys.exit(1)
        elif data_type == "sleep":
            from pre_process.load_sleep_data import LoadSleepData
            from pre_process.pre_process import PreProcess
            import numpy as np

            load_sleep_data = LoadSleepData(data_type="spectrogram")
            pre_process = PreProcess(load_sleep_data)
            datas = load_sleep_data.load_data(load_all=True)
            (train, test) = pre_process.split_train_test_from_records(
                datas, test_id=test_id, pse_data=pse_data
            )
            (x_train, y_train), (_, _) = pre_process.make_dataset(
                train=train,
                test=test,
                is_storchastic=False,
                to_one_hot_vector=False,
                pse_data=pse_data,
            )
            x_train = np.reshape(x_train, (-1, 128 * 512))
            X_reduced = PCA(n_components=2).fit_transform(y_train)
        else:
            print(f"TODO : {data_type}は自分で実装してね")
    if show_data:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        mappable = ax.scatter(
            X_reduced[:, 0], X_reduced[:, 1], c=digits.target
        )
        plt.title("pca", fontsize=20)
        fig.colorbar(mappable, ax=ax)
        plt.show()
        plt.clf()
    else:
        return X_reduced, digits.target


# t-sne
def make_t_sne(data=None, pse_data=True, path=None, show_data=True):
    if pse_data:
        digits = datasets.load_digits()
        # shapeの確認
        print("digits.data.shape : ", digits.data.shape)
        # shape : (size, dim<=指定した圧縮後の次元)
        X_reduced = TSNE(n_components=2, random_state=0).fit_transform(
            digits.data
        )
        # 圧縮後の次元を確認
        print("shape : ", X_reduced.shape)
    else:
        print("実装してね")
        sys.exit(1)
    if show_data:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        mappable = ax.scatter(
            X_reduced[:, 0], X_reduced[:, 1], c=digits.target
        )
        plt.title("t-sne", fontsize=20)
        fig.colorbar(mappable, ax=ax)
        plt.show()
        plt.clf()

    else:
        return X_reduced, digits.target


def make_parallel_graph(targets, datas, comp_types):
    print("len(datas) : ", len(datas))
    fig = plt.figure()
    for i, (target, data, comp_type) in enumerate(
        zip(targets, datas, comp_types)
    ):
        # 表示は(1, len(data))の形
        ax = fig.add_subplot(1, len(datas), i + 1)
        # 最後の時はカラーマップを入れる
        # if i+1==len(targets):
        # colorbarを表示するためにmappableに入れる
        mappable = ax.scatter(data[:, 0], data[:, 1], c=target)
        fig.colorbar(mappable, ax=ax)
        plt.title(comp_type, fontsize=20)
        # else:
        # mappable = ax.scatter(data[:, 0], data[:, 1])
    plt.show()


# 5段階=>4段階推定を行う関数(NR12を統合)
def calc_4ss_from_5ss(ss_df: DataFrame) -> DataFrame:
    # NR2(1) => NR1(2)
    ss_df["y_true"][ss_df["y_true"] == 1] = 2
    ss_df["y_pred_main"][ss_df["y_pred_main"] == 1] = 2
    ss_df["y_pred_sub"][ss_df["y_pred_sub"] == 1] = 2
    # NR34(0) => NR34(1)
    ss_df["y_true"][ss_df["y_true"] == 0] = 1
    ss_df["y_pred_main"][ss_df["y_pred_main"] == 0] = 1
    ss_df["y_pred_sub"][ss_df["y_pred_sub"] == 0] = 1
    # 現在存在する睡眠段階
    # Wake: 4, Rem: 3, NR12: 2, NR34: 1
    return ss_df


# csvファイルから一致率を計算
def calc_acc_based_on_unc(df: DataFrame) -> tuple:
    unc_threthold = [0.1 * i for i in range(1, 11)]
    acc_list = list()
    model_list = list()
    unc_list = list()
    for unc in unc_threthold:
        df_trimmed = df[df["unc_base"] < unc]
        # マージした予測ラベル
        # NOTE: iterrowsメソッドにて行のループ処理可能
        y_pred_merged_series = [
            rec["y_pred_main"] if rec["unc_base"] < 0.5 else rec["y_pred_sub"]
            for _, rec in df_trimmed.iterrows()
        ]
        # 全体のラベル数
        all_num = df_trimmed.shape[0]
        # 正解のラベル数
        true_num = sum(df_trimmed["y_true"] == y_pred_merged_series)
        true_num_base = sum(df_trimmed["y_true"] == df_trimmed["y_pred_main"])
        # 一致率
        acc = true_num / (all_num + 0.00001)
        acc_base = true_num_base / (all_num + 0.00001)
        acc_list.extend([acc, acc_base])
        unc_list.extend([unc, unc])
        model_list.extend(["Proposed", "base"])

    return unc_list, acc_list, model_list


if __name__ == "__main__":
    import os
    from glob import glob
    import pandas as pd
    from tqdm import tqdm

    folder_path = os.path.join(
        os.environ["sleep"], "tmp_outputs", "enn_outputs_4stage"
    )
    target_files = glob(folder_path + "/*.csv")
    # 一致率,uncのリスト
    acc_list = list()
    model_list = list()
    unc_list = list()
    # 読み込むファイルでループ処理
    for filepath in tqdm(target_files):
        df = pd.read_csv(filepath)
        unc, acc, model = calc_acc_based_on_unc(df)
        acc_list.extend(acc)
        unc_list.extend(unc)
        model_list.extend(model)

    df = pd.DataFrame(
        {"unc": unc_list, "accuracy": acc_list, "model": model_list}
    )
    folder_path = os.path.join(
        os.environ["sleep"], "tmp_outputs", "enn_outputs_4stage_accuracy"
    )
    filepath = os.path.join(folder_path, "output.csv")
    df.to_csv(filepath)
