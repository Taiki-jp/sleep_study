from typing import Any
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.ops.numpy_ops.np_arrays import ndarray
from data_analysis.py_color import PyColor
import pickle
import datetime
import os
import wandb
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
import numpy as np
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

    def stop_early(self, mode: str, *args: tuple):
        if mode == "catching_assertion":
            try:
                assert args[0].shape[0] != 0 or args[1].shape[1] != 0
            except:
                # 本当は止めずに、このループの処理だけ飛ばして続けたい
                raise AssertionError("データを拾えませんでした。止めます")
        else:
            raise Exception("知らないモードが指定されました")

    # graph_person_id => test_label(test_name), graph_date_id => date_id
    def make_graphs(
        self,
        y: ndarray,
        evidence: Tensor,
        train_or_test: str,
        graph_person_id: str,
        graph_date_id: str,
        calling_graph: Any,
    ):
        if calling_graph == "all":
            # 混合行列をwandbに送信
            self.conf_mat2Wandb(
                y=y,
                evidence=evidence,
                train_or_test=train_or_test,
                test_label=graph_person_id,
                date_id=graph_date_id,
            )
            for is_separating in [True, False]:
                self.u_hist2Wandb(
                    y=y,
                    evidence=evidence,
                    train_or_test=train_or_test,
                    test_label=graph_person_id,
                    date_id=graph_date_id,
                    separate_each_ss=is_separating,
                )
            self.u_threshold_and_acc2Wandb(
                y=y,
                evidence=evidence,
                train_or_test=train_or_test,
                test_label=graph_person_id,
                date_id=graph_date_id,
            )

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
        if not os.path.exists(path):
            print(PyColor.RED_FLASH, f"{path}を作成します", PyColor.END)
            os.makedirs(path)
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
        plt.close()
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
        plt.close()
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
            plt.close()

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
    # FIXME: 削除予定
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
            # 分類数5で正解・予測がともに4クラスしかないときはNR34を取り除く（片方が5クラスあれば大丈夫）
            if (
                len(Counter(y_true)) == 4
                and len(Counter(y_pred)) == 4
                and min(y_true) == 1
                and min(y_pred) == 1
            ):
                # 分類数5で正解・予測がともに3クラスしかないときはNR1を取り除く（片方が5クラスあれば大丈夫）
                labels.pop(0)

        elif n_class == 4:
            labels = ["nr34", "nr12", "rem", "wake"]
        elif n_class == 3:
            labels = ["nrem", "rem", "wake"]
        elif n_class == 2:
            labels = ["non-target", "target"]
        df = pd.DataFrame(cm)
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
        colors = list(color_obj.__dict__.values())[11 : 11 + target_len]
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
    def conf_mat2Wandb(
        self,
        y,
        evidence,
        train_or_test,
        test_label,
        date_id,
        log_all_in_one: bool = False,
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
        y: Tensor,
        evidence: Tensor,
        train_or_test: str,
        test_label: str,
        date_id: str,
        dir2: str = "histgram",
        separate_each_ss: bool = False,
        unc: Tensor = None,
        has_caliculated: bool = False,
        alpha: Tensor = None,
        y_pred: Tensor = None,
        log_all_in_one: bool = False,
    ):
        # 計算済みの場合はそれを使うほうが良い
        if has_caliculated:
            try:
                assert (
                    evidence is not None
                    and unc is not None
                    and alpha is not None
                    and y_pred is not None
                )
            except AssertionError:
                print(
                    PyColor.RED_FLASH,
                    "計算済みの場合，evidence, unc, alpha, y_pred を渡してください",
                    PyColor.END,
                )
                sys.exit(1)

        else:
            # 各睡眠段階に分けて表示するかどうか
            alpha = evidence + 1
            S = tf.reduce_sum(alpha, axis=1, keepdims=True)
            y_pred = alpha / S
            _, n_class = y_pred.shape
            # 今は5クラス分類以外ありえない
            assert n_class == 5
            # カテゴリカルに変換
            y_pred = (
                np.argmax(y_pred, axis=1)
                if not self.catch_nrem2
                else self.my_argmax(y_pred, axis=1)
            )
        # unc だけを渡す場合
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
        self,
        y: ndarray,
        evidence: Tensor,
        test_label: str,
        train_or_test: str,
        date_id: str,
        log_all_in_one: bool = False,
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

        if log_all_in_one:
            # 被験者すべてに関して同じグラフにまとめるためにwandbに送信
            data = [
                [__unc4wandb, __acc4wandb]
                for (__unc4wandb, __acc4wandb) in zip(
                    thresh_hold_list, acc_list
                )
            ]
            table = wandb.Table(
                data=data, columns=["unc_threthold", "accuracy"]
            )
            wandb.log(
                {
                    "unc-acc_plot": wandb.plot.line(
                        table,
                        "unc_threthold",
                        "accuracy",
                        title="unc-acc plot",
                    )
                }
            )

        # グラフの作成
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        # 五角形のプロット(一致率)
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
        # 三角形のプロット(正解数)
        ax2.scatter(
            _thresh_hold_list,
            true_list,
            c="#43caf4",
            label="true_num",
            marker="^",
        )
        # なぞる
        ax2.plot(_thresh_hold_list, true_list, c="#43caf4", linestyle=":")
        # 四角形のプロット(全体のサンプル数)
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
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(path, "uncertainty" + "_" + date_id + ".png")
        plt.savefig(file_path)
        # 保存したグラフをwandbに送信
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

    # 極座標で考える
    def polar_data(
        self, row: int, col: int, x_bias: int, y_bias: int
    ) -> tuple:
        r_class0 = tf.random.uniform(shape=(row,), minval=0, maxval=0.6)
        theta_class0 = tf.random.uniform(
            shape=(row,), minval=0, maxval=np.pi * 2
        )
        r_class1 = tf.random.uniform(shape=(row,), minval=0.5, maxval=1)
        theta_class1 = tf.random.uniform(
            shape=(row,), minval=0, maxval=np.pi * 2
        )
        input_class0 = (
            x_bias + r_class0 * np.cos(theta_class0),
            y_bias + r_class0 * np.sin(theta_class0),
        )
        input_class1 = (
            x_bias + r_class1 * np.cos(theta_class1),
            y_bias + r_class1 * np.sin(theta_class1),
        )
        x_train = tf.concat([input_class0, input_class1], axis=1)
        x_train = tf.transpose(x_train)
        y_train_0 = [0 for _ in range(row)]
        y_train_1 = [1 for _ in range(row)]
        y_train = y_train_0 + y_train_1
        x_test = None
        y_test = None
        return (x_train, x_test), (y_train, y_test)

    # 点対称なデータセット
    def point_symmetry_data(
        self, row: int, col: int, x_bias: int, y_bias: int, seed: int = 0
    ) -> tuple:
        datas = tf.random.uniform(
            shape=(row * 2, col),
            minval=-1,
            maxval=1,
            seed=seed,
        )
        # label 分け
        labels = [0 if data[0] * data[1] >= 0 else 1 for data in datas]
        # test データは用意しない
        return (datas, None), (labels, None)

    # アルキメデスの渦巻線
    def archimedes_spiral(
        self, row: int, col: int, x_bias: int, y_bias: int, seed: int = 0
    ) -> tuple:
        theta_class0 = tf.random.uniform(
            shape=(row,), minval=0, maxval=np.pi * 4
        )
        theta_class1 = tf.random.uniform(
            shape=(row,), minval=0, maxval=np.pi * 4
        )
        input_class0 = (
            x_bias + (theta_class0 / 4 / np.pi) * np.cos(theta_class0),
            y_bias + (theta_class0 / 4 / np.pi) * np.sin(theta_class0),
        )
        # pi ずらす
        input_class1 = (
            x_bias + np.cos(theta_class1 + np.pi) * (theta_class1 / 4 / np.pi),
            y_bias + np.sin(theta_class1 + np.pi) * (theta_class1 / 4 / np.pi),
        )
        x_train = tf.concat([input_class0, input_class1], axis=1)
        x_train = tf.transpose(x_train)
        y_train_0 = [0 for _ in range(row)]
        y_train_1 = [1 for _ in range(row)]
        y_train = y_train_0 + y_train_1
        x_test = None
        y_test = None
        return (x_train, x_test), (y_train, y_test)

    # 仮データ作成メソッドの親メソッド
    def make_2d_psedo_data(
        self,
        row: int,
        col: int,
        x_bias: int = 0,
        y_bias: int = 0,
        seed: int = 0,
        data_type: str = "",
    ) -> tuple:
        # 極座標のデータ
        if len(data_type) == 0 and type(data_type) == str:
            print(PyColor.RED_FLASH, "データタイプを指定してください", PyColor.END)
        elif data_type == "type01":
            return self.polar_data(
                row=row, col=col, x_bias=x_bias, y_bias=y_bias
            )
        elif data_type == "type02":
            return self.point_symmetry_data(row, col, x_bias, y_bias, seed)
        elif data_type == "type03":
            return self.archimedes_spiral(row, col, x_bias, y_bias, seed)
        else:
            print(PyColor.RED_FLASH, "データタイプの指定方法を確認してください", PyColor.END)
            sys.exit(1)


if __name__ == "__main__":
    utils = Utils()
    (x_train, x_test), (y_train, y_test) = utils.archimedes_spiral(
        100, 2, 0, 0
    )
    x_train = x_train.numpy()
    plt.scatter(x_train[:100, 0], x_train[:100, 1], c="r")
    plt.scatter(x_train[100:, 0], x_train[100:, 1], c="b")
    plt.savefig("hoge.png")

    # ===============
    # make graph test
    # ===============
    # root_dir = os.path.join(os.environ["sleep"], "figures")
    # each_dir_name_list = ["main_network", "sub_network", "merged_network"]
    # saved_path_list = [
    #     os.path.join(
    #         root_dir, each_dir_name_list[i], "check_uncertainty", "16", "8"
    #     )
    #     for i in range(3)
    # ]
    # for saved_path in saved_path_list:
    #     utils.make_gif(saved_path=saved_path)

    # =========================
    # point_symmetry_data test
    # =========================
    # (x_train, x_test), (y_train, y_test) = utils.polar_data(100, 2, 0, 0)
    # print(x_train)
    # =========================
    # archimedes_spiral test
    # =========================
