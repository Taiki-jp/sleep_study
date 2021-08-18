from data_analysis.py_color import PyColor
import pickle, datetime, os, wandb
from pre_process.file_reader import FileReader
from random import shuffle, choices, random, seed
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
from tensorflow.keras.datasets import mnist
from pre_process.load_sleep_data import LoadSleepData
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from collections import Counter
from IPython.display import SVG

# from keras.utils.vis_utils import model_to_dot
import sys
import numpy
import tensorflow as tf
from data_analysis.my_color import MyColor
from pre_process.my_env import MyEnv
from PIL import Image
import glob

# 便利な関数をまとめたもの
class Utils:
    def __init__(self, ss_list=None, catch_nrem2: bool = False) -> None:
        self.id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.fr = FileReader()
        self.env = self.fr.my_env
        # self.name_list = self.fr.sl.name_list
        # self.name_dict = self.fr.sl.name_dict
        self.ss_list = ss_list
        self.catch_nrem2 = catch_nrem2

    def dump_with_pickle(self, data, file_name, data_type, fit_pos):

        file_path = os.path.join(
            self.env.pre_processed_dir, data_type, fit_pos
        )
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join(file_path, file_name + "_" + self.id + ".sav")
        print(PyColor.CYAN, PyColor.BOLD, f"{file_path}を保存します", PyColor.END)
        pickle.dump(data, open(file_path, "wb"))

    def showSpectrogram(self, *datas, num=4, path=False):
        """正規化後と正規化前の複数を同時にプロットするためのメソッド
        例
        >>> o_utils.showSpectrogram(x_train, tmp)
        このとき x_trian.shape=(batch, row(128), col(512)) となっている（tmp　も同様）
        NOTE : 周波数を行方向に取るために転置をプログラム内で行っている(data[k].T のところ)

        Args:
            num (int, optional): [繰り返したい回数]. Defaults to 4.
            path (bool, optional): [保存場所を指定したいときはパスを渡す．デフォルトではsleep_study/figures/tmp に入る]. Defaults to False.
        """
        fig = plt.figure()
        for i, data in enumerate(datas):
            for k in range(num):
                ax = fig.add_subplot(len(datas), num, 1 + k + num * i)
                im = ax.imshow(data[k].T)
                cbar = fig.colorbar(im)
        if path:
            plt.savefig(path)
        else:
            new_path = os.path.join(self.defaultPath, self.id + ".png")
            plt.savefig(new_path)
        plt.clf()

    # accuracy, precision, recall, f-measureを計算する
    def compute_properties(self, df):
        """NOTE : df はwandbの形式に合わせて入っているものとする"""
        tp = df[1][0]
        fp = df[0][0]
        tn = df[0][1]
        fn = df[1][1]
        try:
            acc = (tp + tn) / (tp + fp + tn + fn)  # 一致率
        except:
            print("accuracyを計算できません", "Noneを返します")
            acc = None
        try:
            pre = (tp) / (tp + fp)  # 予測の適合度
        except:
            print("presicionを計算できません", "Noneを返します")
            pre = None
        try:
            rec = (tp) / (tp + fn)  # 実際のうちの再現度
        except:
            print("recallを計算できません", "Noneを返します")
            rec = None
        f_m = 2 * tp / (2 * tp + fp + fn)
        return acc, pre, rec, f_m

    # 被験者ごとに合わせて評価値を表示する
    def show_properties(self, df_list):
        # df_list はlist型の9x5の構造になっているようにする
        assert type(df_list) == list
        assert df_list.__len__() == 9
        assert df_list[0].__len__() == 5
        returned_dict = dict()
        for _df_list, name in zip(df_list, self.name_list):
            for df, ss in zip(_df_list, self.ss_list):
                dict_key = name + "_" + ss
                assert df.shape == (2, 2)
                returned_dict.update({dict_key: self.compute_properties(df)})
        return returned_dict

    # グラフを並べて表示する
    def plot_images(self, images_arr):
        fig, axes = plt.subplots(1, 5, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    # wandbに画像のログを送る
    def save_image2Wandb(
        self,
        image,
        dir2="confusion_matrix",
        fileName="cm",
        to_wandb=False,
        train_or_test=None,
        test_label=None,
        date_id=None,
    ):
        sns.heatmap(image, annot=True, cmap="Blues", fmt="d")
        plt.xlabel("pred")
        plt.ylabel("actual")
        # 保存するフォルダ名を取得
        path = os.path.join(
            self.env.figure_dir, dir2, test_label, train_or_test
        )
        # パスを通す
        self.check_path_auto(path=path)
        # ファイル名を指定して保存
        plt.savefig(os.path.join(path, fileName + "_" + date_id + ".png"))

        if to_wandb:
            im_read = plt.imread(
                os.path.join(path, fileName + "_" + date_id + ".png")
            )
            wandb.log(
                {
                    f"{dir2}:{train_or_test}": [
                        wandb.Image(im_read, caption=f"{fileName}")
                    ]
                }
            )
        plt.clf()
        return

    # wandbにグラフのログを送る
    def save_graph2Wandb(self, path, name, train_or_test):
        im_read = plt.imread(path)
        # 画像送信
        wandb.log(
            {
                f"{name}:{train_or_test}": [
                    wandb.Image(im_read, caption=f"{name}")
                ]
            }
        )
        # 画像削除
        plt.clf()
        return

    # attention, imageを並べて表示する
    def simpleImage(
        self,
        image_array,
        row_image_array,
        file_path,
        x_label=None,
        y_label=None,
        title_array=None,
    ):
        assert image_array.ndim == 4 and row_image_array.ndim == 3
        for num, image in enumerate(image_array):
            fig = plt.figure(121)
            ax1 = fig.add_subplot(121)
            ax1.axis("off")
            im1 = ax1.imshow(image[:, :, 0].T)
            cbar1 = fig.colorbar(im1)
            ax2 = fig.add_subplot(122)
            # ax2.axis('off')  # input の方はメモリを入れる
            im2 = ax2.imshow(row_image_array[num].T, aspect="auto")
            cbar2 = fig.colorbar(im2)
            ax1.set_title("Attention")
            ax2.set_title("Input")
            ax2.set_xticks(np.arange(0, 128 + 1, 32))
            ax2.set_xticklabels(np.arange(0, 32 + 1, 8))
            ax2.set_yticks(np.arange(0, 512 + 1, 64))
            ax2.set_yticklabels(np.arange(0, 8 + 1, 1))
            if x_label:
                ax1.set_xlabel(x_label)
                ax2.set_xlabel(x_label)
            if y_label:
                ax1.set_ylabel(y_label)
                ax2.set_ylabel(y_label)
            plt.tight_layout()
            plt.suptitle(f"conf : {title_array[num].numpy():.0%}", size=10)
            plt.savefig(os.path.join(file_path, f"{num}"))
            plt.clf()

    # ネットワークグラフを可視化する
    def makeNetworkGraph(self, model, dir2="test", fileName="test"):
        path = self.env.figure_dir + os.path.join(dir2, fileName + ".png")
        SVG(data=model_to_dot(model).create(prog="dot", format="svg"))

    # パスの存在を確認をして無ければ作成する
    def check_path(self, path):
        if not os.path.exists(path):
            print(f"フォルダが見つからなかったので以下の場所に作成します．\n {path}")
            os.mkdir(path)

    # 存在しないディレクトリを自動で作成する
    def check_path_auto(self, path):
        dir_list = path.split("\\")
        _dir_list = dir_list
        # dir_listの最後にはフォルダではなくファイル名が来ている想定
        file_path = ""
        # FIXME : dir_list の開始地点はインデックスが5以上のTaikiSenjuの下にする
        # NOTE : root_dirを睡眠のディレクトリとすることでその下を確認する
        _, root_dir_name = os.path.split(self.env.project_dir)
        for _ in dir_list:
            if dir_list[0] == root_dir_name:
                # 先頭要素をpop（睡眠のディレクトリまでpop）
                dir_list.pop(0)
                break
            # 先頭要素をpop
            dir_list.pop(0)

        # 以下のforループで更新対象の文字列（パス）
        file_path = self.env.project_dir

        # 上のpopで生き残った（睡眠のディレクトリ以下）ディレクトリの存在チェック
        for dir_name in dir_list:
            file_path = os.path.join(file_path, dir_name)
            self.check_path(file_path)

    # 混合行列を作成
    def make_confusion_matrix(self, y_true, y_pred, n_class=5):
        # カテゴリカルのデータ(y_true, y_pred)であることを想定
        try:
            assert np.ndim(y_true) == 1
        except:
            print("正解データはlogitsで入力してください（one-hotじゃない形で！）")
            sys.exit(1)
        try:
            assert np.ndim(y_pred) == 1
        except:
            print("予測ラベルはlogitsで入力してください（one-hotじゃない形で！）")

        # 混合行列を作成
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

        # 予測ラベルのクラス数に応じてラベル名を変更する
        if n_class == 5:
            labels = ["nr34", "nr2", "nr1", "rem", "wake"]
        elif n_class == 4:
            labels = ["nr34", "nr12", "rem", "wake"]
        elif n_class == 3:
            labels = ["nrem", "rem", "wake"]
        elif n_class == 2:
            labels = ["non-target", "target"]
        df = pd.DataFrame(cm, columns=labels, index=labels)
        return df

    def makeConfusionMatrixFromInput(self, x, y, model, using_pandas=False):
        """混合マトリクスを作成するメソッド

        Args:
            x ([array]]): [入力データ]
            y ([array]]): [正解ラベル]
            model ([tf.keras.Model]]): [NN モデル]
        """
        pred = model.predict(x)
        cm = confusion_matrix(np.argmax(pred, axis=1), y)
        if using_pandas:
            try:
                df = pd.DataFrame(
                    cm,
                    index=["wake", "rem", "nr1", "nr2", "nr34"],
                    columns=["wake", "rem", "nr1", "nr2", "nr34"],
                )
            except:
                df = pd.DataFrame(cm)
            return cm, df
        return cm, None

    # ヒストグラムの作成・保存のひな形（渡したtarget_arrayの分ヒストグラムを作成）
    def _make_histgram(
        self,
        target_array: list,
        dir2: str,
        file_name: str,
        train_or_test: bool = False,
        test_label: str = None,
        date_id: str = None,
        hist_label: list = None,
        axis_label: dict = {"x": "uncertainty", "y": "samples"},
        color_obj: MyColor = MyColor(),
    ):
        plt.style.use("default")
        sns.set()
        sns.set_style("whitegrid")
        sns.set_palette("Set1")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        target_len = len(target_array)
        try:
            assert len(color_obj.__dict__) > target_len
        except:
            print("your color is smaller than target array")
            sys.exit(1)
        colors = list(color_obj.__dict__.values())[2 : 2 + target_len]
        ax.hist(
            target_array, bins=10, label=hist_label, stacked=True, color=colors
        )
        ax.set_xlabel(axis_label["x"])
        ax.set_ylabel(axis_label["y"])
        plt.legend()
        # 保存するフォルダ名を取得
        path = os.path.join(
            self.env.figure_dir, dir2, test_label, train_or_test
        )
        if not os.path.exists(path):
            os.makedirs(path)
        # 保存
        plt.savefig(os.path.join(path, file_name + "_" + date_id + ".png"))

    # ヒストグラムを作成・保存
    def make_histgram_true_or_false(
        self,
        true_label_array: list = None,
        false_label_array: list = None,
        dir2: str = "histgram",
        file_name: str = "hist",
        train_or_test: list = None,
        test_label: str = None,
        date_id: str = None,
        hist_label: list = ["True", "False"],
        axis_label: dict = {"x": "uncertainty", "y": "samples"},
        color_obj: MyColor = MyColor(),
    ):

        self._make_histgram(
            target_array=[true_label_array, false_label_array],
            dir2=dir2,
            file_name=file_name,
            train_or_test=train_or_test,
            test_label=test_label,
            date_id=date_id,
            hist_label=hist_label,
            axis_label=axis_label,
            color_obj=color_obj,
        )

    # NOTE: 5クラス分類用の設定になっている
    def my_argmax(self, array: np.ndarray, axis: int) -> np.ndarray:
        array_max = np.argmax(array, axis=axis)
        array_min = np.argmin(array, axis=axis)
        fixed_array = []
        # 最大値と最小値が一致する場合はNREM2(1)を返す
        for _max, _min in zip(array_max, array_min):
            if _max == _min:
                fixed_array.append(1)
            else:
                fixed_array.append(_max)
        return np.array(fixed_array)

    # 混合行列をwandbに送信
    def conf_mat2Wandb(self, y, evidence, train_or_test, test_label, date_id):
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        y_pred = alpha / S
        _, n_class = y_pred.shape
        # カテゴリカルに変換
        y_pred = (
            np.argmax(y_pred, axis=1)
            if not self.catch_nrem2
            else self.my_argmax(y_pred, axis=1)
        )
        # ラベル付き混合行列を返す
        cm = self.make_confusion_matrix(
            y_true=y, y_pred=y_pred, n_class=n_class
        )
        # seabornを使ってグラフを作成し保存
        self.save_image2Wandb(
            image=cm,
            to_wandb=True,
            train_or_test=train_or_test,
            test_label=test_label,
            date_id=date_id,
        )
        return

    # 不確かさのヒストグラムをwandbに送信
    def u_hist2Wandb(
        self,
        y,
        evidence,
        train_or_test,
        test_label,
        date_id,
        dir2="histgram",
        separate_each_ss=False,
        unc=None,
    ):
        # 各睡眠段階に分けて表示するかどうか
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        y_pred = alpha / S
        _, n_class = y_pred.shape
        # カテゴリカルに変換
        y_pred = (
            np.argmax(y_pred, axis=1)
            if not self.catch_nrem2
            else self.my_argmax(y_pred, axis=1)
        )
        if unc is not None:
            uncertainty = unc
        else:
            uncertainty = n_class / tf.reduce_sum(alpha, axis=1, keepdims=True)

        if not separate_each_ss:
            # true_label, false_labelに分類する
            true_label_array = uncertainty.numpy()[y == y_pred]
            false_label_array = uncertainty.numpy()[y != y_pred]
            true_label_array = [label[0] for label in true_label_array]
            false_label_array = [label[0] for label in false_label_array]
            # ヒストグラムを作成
            self.make_histgram_true_or_false(
                true_label_array=true_label_array,
                false_label_array=false_label_array,
                train_or_test=train_or_test,
                test_label=test_label,
                date_id=date_id,
            )

        else:
            # 各睡眠段階に分割する
            nrem34_list = uncertainty.numpy()[y == 0]
            nrem2_list = uncertainty.numpy()[y == 1]
            nrem1_list = uncertainty.numpy()[y == 2]
            rem_list = uncertainty.numpy()[y == 3]
            wake_list = uncertainty.numpy()[y == 4]
            nrem34_list = [nrem34[0] for nrem34 in nrem34_list]
            nrem2_list = [nrem2[0] for nrem2 in nrem2_list]
            nrem1_list = [nrem1[0] for nrem1 in nrem1_list]
            rem_list = [rem[0] for rem in rem_list]
            wake_list = [wake[0] for wake in wake_list]

            ss_list = [
                nrem34_list,
                nrem2_list,
                nrem1_list,
                rem_list,
                wake_list,
            ]

            hist_label = ["nrem34", "nrem2", "nrem1", "rem", "wake"]

            # ヒストグラムを作成
            self._make_histgram(
                target_array=ss_list,
                dir2="histgram",
                file_name="hist",
                train_or_test=train_or_test,
                test_label=test_label,
                date_id=date_id,
                hist_label=hist_label,
            )

        file_path = os.path.join(
            self.env.figure_dir,
            dir2,
            test_label,
            train_or_test,
            "hist" + "_" + date_id + ".png",
        )
        # wandbに送信
        self.save_graph2Wandb(
            path=file_path, name="hist", train_or_test=train_or_test
        )

    # 閾値を設定して分類した時の一致率とサンプル数をwandbに送信
    def u_threshold_and_acc2Wandb(
        self, y, evidence, test_label, train_or_test, date_id
    ):
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        y_pred = alpha / S
        _, n_class = y_pred.shape
        # カテゴリカルに変換
        y_pred = (
            np.argmax(y_pred, axis=1)
            if not self.catch_nrem2
            else self.my_argmax(y_pred, axis=1)
        )
        uncertainty = n_class / tf.reduce_sum(alpha, axis=1, keepdims=True)
        # 1次元に落とし込む
        uncertainty = tf.reshape(uncertainty, -1)
        # listに落とし込む
        uncertainty = [u.numpy() for u in uncertainty]
        # 一致率のリスト
        acc_list = list()
        # 正解数のリスト
        true_list = list()
        # 残ったサンプル数のリスト
        existing_list = list()
        # 閾値の空リスト
        _thresh_hold_list = list()
        # 閾値のリスト
        thresh_hold_list = np.arange(0.1, 1.1, 0.1)
        for thresh_hold in thresh_hold_list:
            tmp_y_pred = y_pred[uncertainty <= thresh_hold]
            tmp_y_true = y[uncertainty <= thresh_hold]
            sum_true = sum(tmp_y_pred == tmp_y_true)
            sum_existing = len(tmp_y_true)
            if sum_existing == 0:
                print("trueラベルがありませんでした")
                continue
            acc = sum_true / sum_existing
            acc_list.append(acc)
            true_list.append(sum_true)
            existing_list.append(sum_existing)
            _thresh_hold_list.append(thresh_hold)

        # グラフの作成
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        # 五角形のプロット
        ax1.scatter(
            _thresh_hold_list,
            acc_list,
            c="#f46d43",
            label="accuracy",
            marker="p",
        )
        # なぞる
        ax1.plot(_thresh_hold_list, acc_list, c="#f46d43", linestyle=":")
        ax1.set_xlabel("uncertainty threshold")
        ax1.set_ylabel("accuracy")
        ax2 = ax1.twinx()
        # 三角形のプロット
        ax2.scatter(
            _thresh_hold_list,
            true_list,
            c="#43caf4",
            label="true_num",
            marker="^",
        )
        # なぞる
        ax2.plot(_thresh_hold_list, true_list, c="#43caf4", linestyle=":")
        # 四角形のプロット
        ax2.scatter(
            _thresh_hold_list,
            existing_list,
            c="#43f4c6",
            label="all_num",
            marker="s",
        )
        # なぞる
        ax2.plot(_thresh_hold_list, existing_list, c="#43f4c6", linestyle=":")
        ax2.set_ylabel("samples")
        ax1.legend()
        ax2.legend()
        plt.legend()
        path = os.path.join(
            self.env.figure_dir, "uncertainty", test_label, train_or_test
        )
        self.check_path_auto(path=path)
        file_path = os.path.join(path, "uncertainty" + "_" + date_id + ".png")
        plt.savefig(file_path)
        self.save_graph2Wandb(
            path=file_path, name="uncertainty", train_or_test=train_or_test
        )
        return

    # gif の作成
    def make_gif(self, saved_path: str):
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        files = glob.glob(os.path.join(saved_path, "*.png"))
        images = list(map(lambda file: Image.open(file), files))
        images[0].save(
            os.path.join(saved_path, "out.gif"),
            save_all=True,
            append_images=images[1:],
            loop=0,
        )


if __name__ == "__main__":
    utils = Utils()
    root_dir = os.path.join(os.environ["sleep"], "figures")
    each_dir_name_list = ["main_network", "sub_network", "merged_network"]
    saved_path_list = [
        os.path.join(
            root_dir, each_dir_name_list[i], "check_uncertainty", "16", "8"
        )
        for i in range(3)
    ]
    for saved_path in saved_path_list:
        utils.make_gif(saved_path=saved_path)
