import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    pca_data, pca_target = make_pca(
        show_data=False, pse_data=True, data_type="sleep"
    )
    t_sne_data, t_sne_target = make_t_sne(show_data=False, pse_data=True)
    compression_types = ("pca", "t-sne")
    make_parallel_graph(
        targets=(pca_target, t_sne_target),
        datas=(pca_data, t_sne_data),
        comp_types=compression_types,
    )
