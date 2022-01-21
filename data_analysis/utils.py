import datetime
import glob
import os
import pickle
import sys
from collections import Counter
from typing import Any, Dict, List

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import PIL
import plotly.express as px
import seaborn as sns
import tensorflow as tf

# from imblearn.over_sampling import SMOTE
from IPython.display import SVG
from PIL import Image

# from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from tensorboard.plugins import projector

# from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.ops.numpy_ops.np_arrays import ndarray

# import tensorflow_docs.vis.embed as embed
import wandb
from data_analysis.my_color import MyColor
from data_analysis.py_color import PyColor
from nn.losses import EDLLoss
from pre_process.my_env import MyEnv

matplotlib.use("Agg")

# 便利な関数をまとめたもの
class Utils:
    def __init__(
        self,
        is_normal,
        is_previous,
        data_type,
        fit_pos,
        stride,
        kernel_size,
        model_type,
        cleansing_type,
        ss_list=None,
        catch_nrem2: bool = False,
    ) -> None:
        self.id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.env = MyEnv(
            is_normal=is_normal,
            is_previous=is_previous,
            data_type=data_type,
            fit_pos=fit_pos,
            stride=stride,
            kernel_size=kernel_size,
            model_type=model_type,
            cleansing_type=cleansing_type,
        )
        self.ss_list = ss_list
        self.catch_nrem2 = catch_nrem2
        self.my_color = MyColor()

    # tensorboard のプロジェクタの作成
    def make_tf_projector(
        self,
        x: Tensor,
        y: ndarray,
        batch_size: int,
        hidden_layer_id: int,
        log_dir: str,
        data_type: str,
        model: Model,
        model_loads: bool = False,
        date_id: str = "",
    ) -> None:
        # モデルの読み込み（コンパイル済み）
        if model_loads and model is None:
            print(
                PyColor().CYAN,
                PyColor().RETURN,
                f"*** {data_type}のモデルを読み込みます ***",
                PyColor().END,
            )
            path = os.path.join(
                os.environ["sleep"], "models", data_type, date_id
            )
            model = tf.keras.models.load_model(
                path, custom_objects={"EDLLoss": EDLLoss(K=5, annealing=0.1)}
            )
            print(
                PyColor().CYAN,
                PyColor().RETURN,
                f"*** {data_type}のモデルを読み込みました ***",
                PyColor().END,
            )
        # 新しいモデルの作成
        new_model = tf.keras.Model(
            inputs=model.input, outputs=model.layers[hidden_layer_id].output
        )
        hidden = new_model.predict(x, batch_size=batch_size)
        hidden = hidden.reshape(x.shape[0], -1)
        evidence = model.predict(x, batch_size=batch_size)
        alpha, _, unc, y_pred = self.calc_enn_output_from_evidence(
            evidence=evidence
        )
        # ディレクトリの作成
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # ラベルの保存
        label_path = os.path.join(log_dir, "metadata.tsv")
        with open(label_path, "w") as f:
            f.write("index\tlabel\tuncrt\n")
            for index, (label, u) in enumerate(zip(y, unc)):
                f.write(f"{index}\t{str(label)}\t{str(u)}\n")
        # チェックポイントの作成
        embedding_var = tf.Variable(hidden, name="hidden")
        check_point_file = os.path.join(log_dir, "embedding.ckpt")
        ckpt = tf.train.Checkpoint(embedding=embedding_var)
        ckpt.save(check_point_file)
        # projectorの設定
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = "metadata.tsv"
        projector.visualize_embeddings(log_dir, config)
        return

    def stop_early(self, mode: str, y: Any) -> bool:
        if mode == "catching_assertion":
            try:
                assert y.shape[0] != 0 or y.shape[1] != 0
                return False
            except:
                # 本当は止めずに、このループの処理だけ飛ばして続けたい
                # raise AssertionError("データを拾えませんでした。止めます")
                return True
        else:
            raise Exception("知らないモードが指定されました")

    # graph_person_id => test_label(test_name), graph_date_id => date_id
    # TODO: train_or_testがstr型で渡すべきなのか，bool型で渡すべきなのかを整理しておく
    def make_graphs(
        self,
        y: ndarray,
        evidence: Tensor,
        train_or_test: str,
        graph_person_id: str,
        graph_date_id: str,
        calling_graph: str,
        evidence_positive: Tensor = None,
        unc_threthold: float = 0,
        is_each_unc: bool = False,
    ):
        if calling_graph == "all":
            # 混合行列をwandbに送信
            self.conf_mat2Wandb(
                y=y,
                evidence=evidence,
                train_or_test=train_or_test,
                test_label=graph_person_id,
                date_id=graph_date_id,
                is_each_unc=is_each_unc,
            )
            for is_separating in [True, False]:
                # 不確かさと正負の関係をヒストグラムにログる
                self.u_hist2Wandb(
                    y=y,
                    evidence=evidence,
                    train_or_test=train_or_test,
                    test_label=graph_person_id,
                    date_id=graph_date_id,
                    separate_each_ss=is_separating,
                )
            # 不確かさによる閾値を設けて一致率を計算
            self.u_threshold_and_acc2Wandb(
                y=y,
                evidence=evidence,
                train_or_test=train_or_test,
                test_label=graph_person_id,
                date_id=graph_date_id,
                unc_threthold=unc_threthold,
                evidence_positive=evidence_positive,
            )
            # 全時間のスペクトログラムをログに送る
            # self.make_time_series_prediction(y=y, evidence=evidence)
        else:
            print("全てのグラフを作成する引数'all'を指定してください")

    def make_time_series_prediction(
        self, y: ndarray, evidence: tf.Tensor
    ) -> None:
        _, _, _, y_pred = self.calc_enn_output_from_evidence(evidence=evidence)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.plot(y)
        ax.plot(y_pred)

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
    def plot_images(
        self, images_arr: np.ndarray, title_arr: np.ndarray, num_plot: int
    ):
        fig, axes = plt.subplots(5, num_plot, figsize=(16, 16))
        axes = axes.flatten()
        for img, title, ax in zip(images_arr, title_arr, axes):
            ax.imshow(img)
            ax.axis("off")
            # ax.set_title(f"ss:{title}")
        plt.tight_layout()
        plt.show()

    # グラフを並べて表示する
    def plot_ss_images(
        self, images_arr: np.ndarray, title_arr: np.ndarray, num_plot: int
    ):
        wake_arr = images_arr[title_arr == 4]
        rem_arr = images_arr[title_arr == 3]
        nrem1_arr = images_arr[title_arr == 2]
        nrem2_arr = images_arr[title_arr == 1]
        nrem34_arr = images_arr[title_arr == 0]
        row_num = 5
        fig, axes = plt.subplots(row_num, num_plot, figsize=(16, 16))
        ss_array = list([wake_arr, rem_arr, nrem1_arr, nrem2_arr, nrem34_arr])
        for row, images in enumerate(ss_array):
            for col in range(num_plot):
                try:
                    axes[row, col].imshow(images[col])

                    axes[row, col].axis("off")
                except:
                    continue
        plt.tight_layout()
        plt.show()

    # グラフをplotlyで表示
    def plotly_images(
        self, images_arr: np.ndarray, title_arr: np.ndarray, num_plot: int
    ):
        wake_arr = images_arr[title_arr == 4]
        rem_arr = images_arr[title_arr == 3]
        nrem1_arr = images_arr[title_arr == 2]
        nrem2_arr = images_arr[title_arr == 1]
        nrem34_arr = images_arr[title_arr == 0]
        fig = px.imshow(wake_arr[0].reshape(-1, 30))
        fig.show()
        # row_num = 5
        # fig, axes = plt.subplots(row_num, num_plot, figsize=(16, 16))
        # ss_array = list([wake_arr, rem_arr, nrem1_arr, nrem2_arr, nrem34_arr])
        # for row, images in enumerate(ss_array):
        #     for col in range(num_plot):
        #         try:
        #             axes[row, col].imshow(images[col])

        #             axes[row, col].axis("off")
        #         except:
        #             continue
        # plt.tight_layout()
        # plt.show()

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
        is_specific_path: bool = False,
        specific_name: str = "",
    ):
        sns.heatmap(image, annot=True, cmap="Blues", fmt="d")
        plt.xlabel("pred")
        plt.ylabel("actual")
        if not is_specific_path:
            # 保存するフォルダ名を取得
            path = os.path.join(
                self.env.figure_dir, dir2, test_label, train_or_test
            )
            # パスを通す
            if not os.path.exists(path):
                print(PyColor.RED_FLASH, f"{path}を作成します", PyColor.END)
                os.makedirs(path)
            # ファイル名を指定して保存
            plt.savefig(os.path.join(path, fileName + "_" + date_id + ".png"))
        else:
            plt.savefig(
                os.path.join(self.env.tmp_dir, specific_name, "cm.png")
            )

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

    # DataFrameをcsv出力する
    def to_csv(self, df: pd.DataFrame, path: str, edit_mode: str):
        if edit_mode == "append":
            df.to_csv(path, mode="a", header=True)
        else:
            print("勝手に実装してください")
            sys.exit(1)

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
        try:
            df = pd.DataFrame(cm, columns=labels, index=labels)
        except:
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
        train_or_test: str,
        test_label: str,
        colors: list,
        date_id: str = "",
        hist_label: list = None,
        axis_label: dict = {"x": "uncertainty", "y": "samples"},
    ) -> None:
        plt.style.use("default")
        sns.set()
        sns.set_style("whitegrid")
        sns.set_palette("Set1")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        target_len = len(target_array)
        ss_list = ["nrem1", "nrem2", "nrem34", "rem", "wake"]
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
        train_or_test: str = "",
        test_label: str = "",
        date_id: str = "",
        hist_label: list = ["True", "False"],
        axis_label: dict = {"x": "uncertainty", "y": "samples"},
    ):
        """[Note that true label array is passed ahead]

        Args:
            true_label_array (list, optional): [description]. Defaults to None.
            false_label_array (list, optional): [description]. Defaults to None.
            dir2 (str, optional): [description]. Defaults to "histgram".
            file_name (str, optional): [description]. Defaults to "hist".
            train_or_test (str, optional): [description]. Defaults to "".
            test_label (str, optional): [description]. Defaults to "".
            date_id (str, optional): [description]. Defaults to "".
            hist_label (list, optional): [description]. Defaults to ["True", "False"].
            axis_label (dict, optional): [description]. Defaults to {"x": "uncertainty", "y": "samples"}.
        """

        self._make_histgram(
            target_array=[true_label_array, false_label_array],
            dir2=dir2,
            file_name=file_name,
            train_or_test=train_or_test,
            test_label=test_label,
            date_id=date_id,
            hist_label=hist_label,
            axis_label=axis_label,
            colors=[
                self.my_color.color[key] for key in ["TRUE_BLUE", "FALSE_RED"]
            ],
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

    # 指定した閾値に応じてデータセットを分離して返すメソッド(ndarrayのとき)
    def separate_unc_data(
        self,
        y_true: ndarray,
        y_pred: ndarray,
        unc: Tensor,
        unc_threthold: float,
    ) -> tuple:
        y_true = y_true[unc <= unc_threthold]
        y_pred = y_pred[unc <= unc_threthold]
        return y_true, y_pred

    # 混合行列をwandbに送信
    def conf_mat2Wandb(
        self,
        y: Tensor,
        evidence: Tensor,
        train_or_test: str,
        test_label: str,
        date_id: str,
        log_all_in_one: bool = False,
        is_each_unc: bool = False,
        n_class: int = 5,
    ):
        evidence, alpha, unc, y_pred = self.calc_enn_output_from_evidence(
            evidence=evidence
        )
        # 不確かさによる閾値に応じて混合マトリクスを作成する
        if is_each_unc:
            unc_threthold = np.arange(0, 1, 0.1)
            for _u_th in unc_threthold:
                (y_true_separated, y_pred_separated) = self.separate_unc_data(
                    y, y_pred, unc, _u_th
                )
                # ラベル付き混合行列を返す
                cm = self.make_confusion_matrix(
                    y_true=y_true_separated,
                    y_pred=y_pred_separated,
                    n_class=n_class,
                )
                # cmが空であればグラフを書かずにループに戻る
                if cm.size == 0:
                    print(
                        PyColor.RED_FLASH,
                        f"{train_or_test}：空のcmが渡されました．プログラムを継続します",
                        PyColor.END,
                    )
                    continue
                # seabornを使ってグラフを作成し保存
                self.save_image2Wandb(
                    image=cm,
                    to_wandb=True,
                    train_or_test=train_or_test,
                    test_label=test_label,
                    date_id=date_id,
                )
        else:
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
        n_class: int = 5,
    ):
        (
            evidence,
            alpha,
            uncertainty,
            y_pred,
        ) = self.calc_enn_output_from_evidence(evidence=evidence)
        uncertainty = np.array(uncertainty)
        # 計算済みの場合はそれを使うほうが良い
        # if has_caliculated:
        #     try:
        #         assert (
        #             evidence is not None
        #             and unc is not None
        #             and alpha is not None
        #             and y_pred is not None
        #         )
        #     except AssertionError:
        #         print(
        #             PyColor.RED_FLASH,
        #             "計算済みの場合，evidence, unc, alpha, y_pred を渡してください",
        #             PyColor.END,
        #         )
        #         sys.exit(1)

        # else:
        #     )
        #     # 各睡眠段階に分けて表示するかどうか
        #     alpha = evidence + 1
        #     S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        #     y_pred = alpha / S
        #     _, n_class = y_pred.shape
        #     # 今は5クラス分類以外ありえない
        #     # カテゴリカルに変換
        #     y_pred = (
        #         np.argmax(y_pred, axis=1)
        #         if not self.catch_nrem2
        #         else self.my_argmax(y_pred, axis=1)
        #     )
        # # unc だけを渡す場合
        # if unc is not None:
        #     uncertainty = unc
        # else:
        #     uncertainty = n_class / tf.reduce_sum(alpha, axis=1, keepdims=True)

        if not separate_each_ss:
            # true_label, false_labelに分類する
            true_label_array = uncertainty[y == y_pred]
            false_label_array = uncertainty[y != y_pred]

            # NOTE: 不確かさをTensor型 or 2次元配列で送るときは下をコメントアウト
            # true_label_array = [label[0] for label in true_label_array]
            # false_label_array = [label[0] for label in false_label_array]

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
            nrem34_list = uncertainty[y == 0]
            nrem2_list = uncertainty[y == 1]
            nrem1_list = uncertainty[y == 2]
            rem_list = uncertainty[y == 3]
            wake_list = uncertainty[y == 4]

            # NOTE: uncertaintyを返す型がtensor型の時は下をコメントアウト（2次元配列っぽくなってるので値を取り出すためにこの処理を入れていた）
            # nrem34_list = [nrem34[0] for nrem34 in nrem34_list]
            # nrem2_list = [nrem2[0] for nrem2 in nrem2_list]
            # nrem1_list = [nrem1[0] for nrem1 in nrem1_list]
            # rem_list = [rem[0] for rem in rem_list]
            # wake_list = [wake[0] for wake in wake_list]

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
                colors=[self.my_color.color[key] for key in hist_label],
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

    # 予測ラベルから一致率の計算
    def calc_acc_from_pred(self, y_true, y_pred, log_label):
        # 一致率の計算
        acc = sum(y_pred == y_true) / len(y_true)
        wandb.log({log_label: acc}, commit=False)
        return acc

    # ENN の計算(evidence => unc, pred, alpha)
    def calc_enn_output_from_evidence(self, evidence: Tensor) -> tuple:
        alpha = evidence + 1
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        y_pred = alpha / S
        _, n_class = y_pred.shape
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
        return evidence, alpha, uncertainty, y_pred

    # 閾値を設定して分類した時の一致率とサンプル数をwandbに送信
    def u_threshold_and_acc2Wandb(
        self,
        y: ndarray,
        evidence: Tensor,
        test_label: str,
        train_or_test: str,
        date_id: str,
        evidence_positive: Tensor = None,
        log_all_in_one: bool = False,
        is_early_stop_and_return_data_frame: bool = False,
        unc_threthold: float = 0,
    ):
        acc_list: List[float] = list()
        true_list: List[float] = list()
        existing_list: List[bool] = list()
        _threshold_list: List[float] = list()
        acc_list_replaced: List[float] = list()
        _threshold_list_replaced: List[float] = list()
        # 閾値のリスト
        thresh_hold_list = np.arange(0.1, 1.1, 0.1)

        # evidence からENNの出力を計算
        _, _, uncertainty, y_pred = self.calc_enn_output_from_evidence(
            evidence=evidence
        )
        # 別のevidence からENNの出力を計算
        if evidence_positive is not None:
            (
                _,
                _,
                uncertainty_replacing,
                y_pred_replacing,
            ) = self.calc_enn_output_from_evidence(evidence=evidence_positive)
            # NOTE: 以下の実装ではベースモデルの不確かさの高い部分（0.5 - 1.0）を置き換えモデルの出力で置き換えるため、順番を気にする必要がある
            # よってサイズの確認による順番が揃っているかどうかのチェックが必要
            try:
                assert len(uncertainty) == len(uncertainty_replacing)
            except:
                raise AssertionError("サイズが揃っていません。データを削ってモデルに入れていませんか？")
        # NOTE
        for thresh_hold in thresh_hold_list:
            # 不確かさが大きい時は別のモデルの予測を使う
            tmp_y_pred = y_pred[uncertainty <= thresh_hold]
            if evidence_positive is not None:
                tmp_y_pred_replacing = y_pred_replacing[
                    uncertainty <= thresh_hold
                ]
            uncertainty = np.array(uncertainty)
            tmp_unc = uncertainty[uncertainty <= thresh_hold]
            # tmp_uncとtmp_y_pred のサイズの確認（順番の確定のために必要）
            if evidence_positive is not None:
                try:
                    assert (
                        tmp_y_pred.shape[0] == tmp_unc.shape[0]
                        and tmp_y_pred_replacing.shape[0]
                        == tmp_y_pred.shape[0]
                    )
                except:
                    raise AssertionError("サイズが揃っていません。原因なんだろう。。")

            if evidence_positive is not None:
                # 不確かさの大きいものは置き換える（0.5以上から不確かなものが入ってくる）
                if thresh_hold > unc_threthold:
                    tmp_y_pred_replaced = [
                        tmp_y_pred[i]
                        if __unc < unc_threthold
                        else tmp_y_pred_replacing[i]
                        for i, __unc in enumerate(tmp_unc)
                    ]
                # 閾値未満であれば空リストを返す
                else:
                    tmp_y_pred_replaced = list()

            tmp_y_true = y[uncertainty <= thresh_hold]

            # 置き換え前の情報
            self.__calc_true_acc(
                pred_labels=tmp_y_pred,
                thresh_hold=thresh_hold,
                true_labels=tmp_y_true,
                acc_list=acc_list,
                true_list=true_list,
                existing_list=existing_list,
                threshold_list=_threshold_list,
                is_replacing_mode=False,
            )

            if evidence_positive is not None:
                # 置き換え後の情報(一致率のみ使用)
                self.__calc_true_acc(
                    pred_labels=tmp_y_pred_replaced,
                    thresh_hold=thresh_hold,
                    true_labels=tmp_y_true,
                    acc_list=acc_list_replaced,
                    threshold_list=_threshold_list_replaced,
                    is_replacing_mode=True,
                )

        if log_all_in_one:
            # 被験者すべてに関して同じグラフにまとめるためにwandbに送信
            data = [
                [__unc4wandb, __acc4wandb, __acc4wandb_replaced]
                for (__unc4wandb, __acc4wandb, __acc4wandb_replaced) in zip(
                    thresh_hold_list, acc_list, acc_list_replaced
                )
            ]
            table = wandb.Table(
                data=data, columns=["unc_threthold", "accuracy", "replaced"]
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
        # csv出力のみ行ってwandbには送信しない
        if is_early_stop_and_return_data_frame:
            d = {
                "accuracy": acc_list,
                "replaced": acc_list_replaced,
                "unc": _threshold_list,
                "unc_replaced": _threshold_list_replaced,
            }
            df = pd.DataFrame.from_dict(d, orient="index")
            # このまま出すと普通の転置が行ってしまうので転置する
            return df.transpose()

        # グラフの作成
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        # 五角形のプロット(一致率)
        ax1.scatter(
            _threshold_list,
            acc_list,
            c="#f46d43",
            label="accuracy",
            marker="p",
        )
        # なぞる
        ax1.plot(_threshold_list, acc_list, c="#f46d43", linestyle=":")
        ax1.set_xlabel("uncertainty threshold")
        ax1.set_ylabel("accuracy")
        ax2 = ax1.twinx()

        # 星形のプロット(一致率)
        ax1.scatter(
            _threshold_list_replaced,
            acc_list_replaced,
            c="#f46d43",
            label="accuracy",
            marker="*",
        )
        # なぞる
        ax1.plot(
            _threshold_list_replaced,
            acc_list_replaced,
            c="#f46d43",
            linestyle=":",
        )
        ax1.set_xlabel("uncertainty threshold")
        ax1.set_ylabel("accuracy_replaced")
        # ax2 = ax1.twinx()

        # 三角形のプロット(正解数)
        ax2.scatter(
            _threshold_list,
            true_list,
            c="#43caf4",
            label="true_num",
            marker="^",
        )
        # なぞる
        ax2.plot(_threshold_list, true_list, c="#43caf4", linestyle=":")
        # 四角形のプロット(全体のサンプル数)
        ax2.scatter(
            _threshold_list,
            existing_list,
            c="#43f4c6",
            label="all_num",
            marker="s",
        )
        # なぞる
        ax2.plot(_threshold_list, existing_list, c="#43f4c6", linestyle=":")
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

    # 正解の数、合計のサンプル数、一致率の計算
    def __calc_true_acc(
        self,
        true_labels: ndarray,
        pred_labels: ndarray,
        thresh_hold: float,
        acc_list: list = list(),
        true_list: list = list(),
        existing_list: list = list(),
        threshold_list: list = list(),
        is_replacing_mode: bool = False,
    ):
        # 空リストが入ってきたら終わり
        if len(pred_labels) == 0:
            print("predラベルがありませんでした")
            return
        sum_true = sum(true_labels == pred_labels)
        sum_existing = len(true_labels)
        # 実ラベルがなかったら終わり
        if sum_existing == 0:
            print("tureラベルがありませんでした")
            return
        acc = sum_true / sum_existing
        acc_list.append(acc)
        threshold_list.append(thresh_hold)
        if is_replacing_mode:
            return
        true_list.append(sum_true)
        existing_list.append(sum_existing)
        return

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
            return None, None
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
            return None, None

    # 各睡眠段階のF値の計算
    def calc_each_ss_f_m(
        self,
        x: Tensor,
        y: Tensor,
        model: Model,
        n_class: int = 5,
        batch_size: int = 32,
    ):
        labels = ["nr34", "nr2", "nr1", "rem", "wake"]
        y_pred = np.argmax(model.predict(x, batch_size=batch_size), axis=1)
        confmatrix = confusion_matrix(
            y_true=y, y_pred=y_pred, labels=range(n_class)
        )
        confdiag = np.eye(len(confmatrix)) * confmatrix
        # print(confdiag)
        np.fill_diagonal(confmatrix, 0)

        # NOTE: Tensor型できたときのみこの処理にする
        if type(y) is not np.ndarray:
            ss_dict: Dict[int, int] = Counter(y.numpy())
        else:
            ss_dict = Counter(y)

        if confmatrix.shape[0] == 5:
            rec_log_dict: Dict[str] = {
                "rec_" + ss_label: confdiag[i][i] / (ss_dict[i])
                if ss_dict[i] != 0
                else np.nan
                for (ss_label, i) in zip(labels, range(len(labels)))
            }
            pre_log_dict = {
                "pre_"
                + ss_label: confdiag[i][i]
                / (sum(confmatrix[i]) + confdiag[i][i])
                if sum(confmatrix[i]) + confdiag[i][i] != 0
                else np.nan
                for (ss_label, i) in zip(labels, range(len(labels)))
            }
            f_m_log_dict = {
                "f_m_" + ss_label: rec * pre * 2 / (rec + pre)
                if rec + pre != 0
                else np.nan
                for (rec, pre, ss_label) in zip(
                    rec_log_dict.values(), pre_log_dict.values(), labels
                )
            }
            rec_log_dict.update(pre_log_dict)
            rec_log_dict.update(f_m_log_dict)
            wandb.log(rec_log_dict, commit=False)
        elif confmatrix.shape[0] == 4:
            # nrem34がないときの処理
            rec_log_dict = {
                "rec_"
                + labels[i + 1]: confdiag[i + 1][i + 1] / (ss_dict[i + 1])
                if ss_dict[i] != 0
                else np.nan
                for i in range(4)
            }
            pre_log_dict = {
                "pre_"
                + labels[i + 1]: confdiag[i + 1][i + 1]
                / sum(confmatrix[i + 1] + confdiag[i + 1][i + 1])
                if sum(confmatrix[i + 1] + confdiag[i + 1][i + 1]) != 0
                else np.nan
                for i in range(4)
            }
            f_m_log_dict = {
                "f_m_" + ss_label: rec * pre * 2 / (rec + pre)
                if rec + pre != 0
                else np.nan
                for (rec, pre, ss_label) in zip(
                    rec_log_dict.values()[1:],
                    pre_log_dict.values()[1:],
                    labels[1:],
                )
            }
            rec_log_dict.update(pre_log_dict)
            rec_log_dict.update(f_m_log_dict)
            wandb.log(rec_log_dict, commit=False)

    # 各睡眠段階の一致率の計算
    def calc_ss_acc(
        self,
        x: Tensor,
        y: Tensor,
        model: Model,
        n_class: int = 5,
        batch_size: int = 32,
        base_or_positive: str = "",
    ):
        if len(base_or_positive) == 0:
            print(
                PyColor.RED_FLASH,
                "baseモデル or positiveモデルを識別する引数base_or_positiveが指定されていません．問題なければここの部分で返さないように指定してください",
                PyColor.END,
            )
            raise Exception("exception has occured")
        # 一致率の計算
        y_pred = self.my_argmax(
            model.predict(x, batch_size=batch_size), axis=1
        )
        acc = sum(y_pred == y.numpy()) / len(y)
        wandb.log({"accuracy_" + base_or_positive: acc}, commit=False)

    def generate_and_save_images(
        self,
        model: tf.keras.Model,
        epoch: int,
        test_sample: tf.Tensor,
        filename: str = "",
    ):
        mean, logvar = model.encode(test_sample)
        z = model.reparameterize(mean, logvar)
        predictions = model.sample(model, z)
        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0], cmap="gray")
            plt.axis("off")

        # tight_layout minimizes the overlap between 2 sub-plots
        file_dir = os.path.join(self.env.figure_dir, "vae")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        if len(filename) == 0:
            filepath = os.path.join(
                file_dir, f"image_at_epoch_{epoch:04d}.png"
            )
        else:
            filepath = os.path.join(file_dir, filename + ".png")

        plt.savefig(filepath)
        plt.close()

    def show_true_image(self, x: tf.Tensor, filename: str = ""):
        fig = plt.figure(figsize=(4, 4))
        for i in range(x.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(x[i, :, :, 0], cmap="gray")
            plt.axis("off")

        file_dir = os.path.join(self.env.figure_dir, "vae")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        if len(filename) == 0:
            filepath = os.path.join(file_dir, f"original_image_of_mine")
        else:
            filepath = os.path.join(file_dir, filename + ".png")
        plt.savefig(filepath)
        plt.show()

    def display_image(self, epoch_no: int, filename: str = ""):
        file_dir = os.path.join(self.env.figure_dir, "vae")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        if len(filename) == 0:
            filepath = os.path.join(
                file_dir, f"image_at_epoch_{epoch_no:04d}.png"
            )
        else:
            filepath = os.path.join(file_dir, filename + ".png")
        plt.savefig(filepath)
        # plt.show()
        return PIL.Image.open(filepath)

    def create_gif(self, outputfile: str = "vdann.gif") -> None:
        file_dir = os.path.join(self.env.figure_dir, "vae")
        outputfile = os.path.join(file_dir, outputfile)
        with imageio.get_writer(outputfile, mode="I") as writer:
            filenames = glob.glob(os.path.join(file_dir, "image*.png"))
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(os.path.join(file_dir, filename))
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)
        embed.embed_file(outputfile)

    def drow_latent_space(
        self,
        model: tf.keras.Model,
        x: tf.Tensor,
        y: tf.Tensor,
        filename: str = "",
    ):
        data = x[::10], y[::10]
        mean, logvar = model.encode(data[0])
        z = model.reparameterize(mean, logvar)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        im = ax.scatter(z[:, 0], z[:, 1], c=data[1])
        cbar = fig.colorbar(im)
        file_dir = os.path.join(self.env.figure_dir, "vae")
        if len(filename) == 0:
            filepath = os.path.join(file_dir, f"vae_latent_space.png")
        else:
            filepath = os.path.join(file_dir, filename + ".png")
        plt.savefig(filepath)

    def compare_ss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        test_name: str,
    ) -> None:
        filepath = os.path.join(self.env.tmp_dir, test_name, "ss.png")
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(311)
        ax1.plot(y_true, label="y_true")
        ax1.legend()
        ax1.set_title(test_name)
        ax2 = fig.add_subplot(312)
        ax2.plot(y_pred, label="y_pred")
        ax2.legend()
        ax3 = fig.add_subplot(313)
        ax3.plot(y_true, label="y_true", alpha=0.4)
        ax3.plot(y_pred, label="y_pred", alpha=0.4)
        ax3.legend()
        if not os.path.exists(os.path.split(filepath)[0]):
            os.makedirs(os.path.split(filepath)[0])
        plt.savefig(filepath)
        plt.close()
        return

    def calc_metrics(self, cm: pd.DataFrame):
        pass

    def calc_unc_under_given_threthold(
        self, unc: List[float], threthold: float
    ) -> None:
        unc = np.array(unc)
        for _threthold in range(10):
            _threthold *= 0.1
            over_unc = len(unc[unc > _threthold])
            wandb.log(
                {"unc_rate" + str(_threthold): over_unc / len(unc)},
                commit=False,
            )
        return


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
