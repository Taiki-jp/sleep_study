import datetime
import itertools
import os
import sys
from collections import Counter
from random import choice, choices, seed, shuffle
from typing import Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.preprocessing.image as tf_image
from imblearn.over_sampling import SMOTE
from numpy import ndarray
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_analysis.py_color import PyColor
from pre_process.load_sleep_data import LoadSleepData, MyEnv, Record

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # tensorflow を読み込む前のタイミングですると効果あり


class PreProcess:
    def __init__(
        self,
        data_type: str,
        fit_pos: str,
        verbose: int,
        kernel_size: int,
        is_previous: bool,
        stride: int,
        is_normal: bool,
        cleansing_type: str,
        has_nrem2_bias: bool = False,
        has_rem_bias: bool = False,
        model_type: str = "",
        make_valdata: bool = False,
        has_ignored: bool = False,
        lsp_option: str = "",
    ):
        self.date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.data_type = data_type
        self.fit_pos = fit_pos
        self.verbose = verbose
        self.kernel_size = kernel_size
        self.is_previous = is_previous
        self.stride = stride
        self.model_type = model_type
        self.is_normal = is_normal
        self.make_valdata = make_valdata
        self.load_sleep_data: LoadSleepData = LoadSleepData(
            data_type=self.data_type,
            fit_pos=self.fit_pos,
            verbose=self.verbose,
            kernel_size=self.kernel_size,
            is_previous=self.is_previous,
            stride=self.stride,
            is_normal=self.is_normal,
            model_type=self.model_type,
            cleansing_type=cleansing_type,
            has_ignored=has_ignored,
            option=lsp_option,
        )
        self.has_nrem2_bias = has_nrem2_bias
        self.has_rem_bias = has_rem_bias
        # LoadSleepData の参照を作成
        self.sl = self.load_sleep_data.sl
        self.my_env: MyEnv = self.load_sleep_data.my_env
        # その他よく使うものをメンバに持っておく
        self.name_list = self.load_sleep_data.name_list
        # self.list2sub method 用の登録辞書とカウンタ
        self.has_catched_name = dict()
        self.subject_counter = 0

    # データセットの作成（この中で出来るだけ正規化なども含めて終わらせる）
    # TODO: データ選択方法の見直し
    # def make_test_data(
    #     self,
    #     test: list,
    #     normalize: bool,
    #     catch_none: bool,
    #     insert_channel_axis: bool,
    #     to_one_hot_vector: bool,
    #     class_size: int,
    #     n_class_converted: int,
    # ):
    #     y_test = self.list2SS(test)
    #     if self.data_type == "spectrum":
    #         x_test = self.list2Spectrum(test)
    #     elif self.data_type == "spectrogram":
    #         x_test = self.list2Spectrogram(test)
    #     elif self.data_type == "cepstrum":
    #         x_test = self.list2Cepstrum(test)
    #     else:
    #         print("spectrum or spectrogram or cepstrumを指定してください")
    #         sys.exit(1)
    #     # Noneの処理をするかどうか
    #     if catch_none is True:
    #         print("- noneの処理を行います")
    #         x_test, y_test = self.catch_none(x_test, y_test)

    #     # max正規化をするかどうか
    #     if normalize is True:
    #         print("- max正規化を行います")
    #         self.max_norm(x_test)

    #     # 睡眠段階のラベルを0 -（クラス数-1）にする
    #     # クラスサイズに合わせて処理を変更する
    #     y_test = self.change_label(
    #         y_data=y_test,
    #         n_class=class_size,
    #         n_class_converted=n_class_converted,
    #     )

    #     # inset_channel_axis based on data type
    #     if self.data_type == "spectrum" or self.data_type == "cepstrum":
    #         if insert_channel_axis:
    #             print("- チャンネル方向に軸を追加します")
    #             x_test = x_test[:, :, np.newaxis]  # .astype('float32')
    #     elif self.data_type == "spectrogram":
    #         if insert_channel_axis:
    #             print("- チャンネル方向に軸を追加します")
    #             x_test = x_test[:, :, :, np.newaxis]  # .astype('float32')

    #     # convert to one-hot vector
    #     if to_one_hot_vector:
    #         print("- one-hotベクトルを出力します")
    #         y_test = tf.one_hot(y_test, class_size)

    #     return (x_test, y_test)

    def make_dataset(
        self,
        catch_none=True,
        class_size=5,
        each_data_size=1000,
        insert_channel_axis=True,
        is_set_data_size=True,
        is_shuffle=True,
        is_storchastic=True,
        is_under_4hz: bool = False,
        n_class_converted: int = 5,
        normalize=True,
        pse_data=False,
        target_ss=None,
        test=None,
        to_one_hot_vector=False,  # NOTE: yの次元は複数の時はone-hotではvstack出来ないのでTrueを使わないように注意
        train=None,
    ) -> Tuple[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray]]:
        # NOTE : when true, make pse_data based on the data type
        # which specified in load_sleep_data object
        if pse_data:
            print("- 仮の睡眠データを作成します")
            return self.make_pse_sleep_data(
                n_class=n_class_converted,  # NOTE: class_sizeの使用を避けるために無理やり買えたので使用の際は要注意
                data_size=each_data_size,
                to_one_hot_vector=to_one_hot_vector,
            )

        if is_set_data_size is False and self.verbose == 0:
            print("データサイズをそろえずにデータセットを作成します")

        if self.verbose == 0:
            print("- 訓練データのサイズを揃えます")

        # 各睡眠段階のサイズを訓練データのために決定する
        ss_dict_train = self.set_datasize(
            class_size=class_size,
            each_data_size=each_data_size,
            is_storchastic=is_storchastic,
        )

        # 補正前の各睡眠段階のクラス数の表示
        if self.verbose == 0:
            print(
                "訓練データの各睡眠段階（補正前）",
                Counter([record.ss for record in train]),
            )

            def _storchastic_sampling(
                data: Dict[int, int],
                target_records: List[Record],
                is_test: bool,
            ) -> Tuple[List[Record], List[Record]]:
                """確率的サンプリングを実際に行うメソッド

                Args:
                    data (Dict[int, int]): キー（睡眠段階），バリュー（サンプル数）を与える
                    target_records (List[Record]): 確率的サンプリング対象のレコードのリスト
                    is_test (bool): 動作テストかどうか

                Returns:
                    Tuple[List[Record], List[Record]]: 訓練データと検証データを返す
                """
                tmp = list()

                # dataで受け取った辞書のバリューで指定された各睡眠段階のリストを返す
                def _splitEachSleepStage() -> List[List[Record]]:
                    each_stage_list = list()

                    for ss in data.keys():
                        # record は順番通りにソートされていない
                        # 多分train自体がソートされたものではないから
                        # data.keys()の順番が5, 2, 4, 3, 1になっているから？
                        # でもその順番に追加されているわけではないから違いそう
                        # なんか最初に4が選択されていた（辞書なので順番がないからそういうものと思う）
                        each_stage_list.append(
                            [
                                record
                                for record in target_records
                                if record.ss == ss
                            ]
                        )
                    return each_stage_list

                # 各睡眠段階がある間はその睡眠段階に対してランダムサンプリングを行う
                def _sample(target_records: List[Record]) -> List[Record]:
                    """各睡眠段階がある間はその睡眠段階に対してランダムサンプリングを行う

                    Args:
                        target_records (List[Record]): ある睡眠段階のみのレコードのリスト型

                    Returns:
                        List[Record]: 重複を許して指定回数ランダム洗濯を繰り返して，レコードのリスト型で返す
                    """
                    if is_test:
                        print(
                            PyColor.RED_FLASH,
                            f"廃止しました。テスト時は{sys._getframe().f_code.co_name}を使用しないようにしてください",
                            PyColor.END,
                        )
                        sys.exit(1)

                    # 睡眠段階のラベルを知るために必要
                    ss = target_records[0].ss
                    # 予測したい睡眠段階の数によって処理を変える
                    if n_class_converted == 5:
                        _selected_list = choices(target_records, k=data[ss])
                    # 予測したい睡眠段階が2段階の時はターゲットクラスの時のみ4倍（つまり他のクラスが4つ存在することを考慮すると同数）
                    elif n_class_converted == 2:
                        ss_int_label_d = {
                            "wake": 5,
                            "rem": 4,
                            "nr1": 3,
                            "nr2": 2,
                            "nr3": 1,
                        }
                        target_ss_as_int = ss_int_label_d[target_ss[0]]
                        if ss != target_ss_as_int:
                            _selected_list = choices(
                                target_records, k=data[ss]
                            )
                        elif ss == target_ss_as_int:
                            # かつnr2(2)であれば、8倍（つまり2倍の量）を作成する
                            if ss == 2:
                                _selected_list = choices(
                                    target_records, k=data[ss] * 8
                                )
                            else:
                                _selected_list = choices(
                                    target_records, k=data[ss] * 4
                                )
                    return _selected_list

                # record を各睡眠段階ごとにわけているもの（長さ５とは限らない（nr3がない場合））
                ss_list = _splitEachSleepStage()

                # 訓練データと検証データを分割するメソッド
                def __split_train_val_data(
                    ratio: Tuple[int, int]
                ) -> Tuple[List[List[Record]], List[List[Record]]]:
                    traindata, valdata = list(), list()
                    for ss_array in ss_list:
                        # シャッフルしてからss_arrayを前半・後半を分ける
                        # NOTE: 時系列データにするときはこの処理を入れられないのでどうしようか
                        shuffle(ss_array)
                        traindata.append(ss_array[: int(0.8 * len(ss_array))])
                        valdata.append(ss_array[int(0.8 * len(ss_array)) :])
                    return traindata, valdata

                # TODO: ここで分割する
                if self.make_valdata:
                    traindata, valdata = __split_train_val_data(ratio=(8, 2))
                else:
                    traindata, valdata = ss_list, None

                # NOTE: サンプリングは訓練データと検証データを分割した後に行う
                tmp = list(map(_sample, traindata))
                # flatten 2d list
                tmp = list(itertools.chain.from_iterable(tmp))
                valdata = list(
                    itertools.chain.from_iterable(valdata)
                )  # 検証データの複製は不必要なので_sample methodを返さなくてよい
                return tmp, valdata

            train, valdata = _storchastic_sampling(
                data=ss_dict_train, target_records=train, is_test=False
            )

            if is_shuffle:
                print("- 訓練データをシャッフルします")
                shuffle(train)

        # TODO : スペクトログラムかスペクトラム化によって呼び出す関数を場合分け
        datasets = (train, valdata, test)
        y_train, y_val, y_test = map(self.list2SS, datasets)
        y_train_subject, y_val_subject, y_test_subject = map(
            self.list2sub, datasets
        )
        if self.data_type == "spectrum":
            converter = self.list2Spectrum
        elif self.data_type == "spectrogram":
            converter = self.list2Spectrogram
        elif self.data_type == "cepstrum":
            converter = self.list2Cepstrum
        else:
            print("spectrum or spectrogram or cepstrumを指定してください")
            sys.exit(1)
        x_train, x_val, x_test = map(converter, datasets)

        # Noneの処理をするかどうか
        if catch_none:
            print("- noneの処理を行います")
            x_train, y_train, y_train_subject = self.catch_none(
                x_train, y_train, y_train_subject
            )
            x_val, y_val, y_val_subject = self.catch_none(
                x_val, y_val, y_val_subject
            )
            x_test, y_test, y_test_subject = self.catch_none(
                x_test, y_test, y_test_subject
            )

        # max正規化をするかどうか
        if normalize:
            print("- max正規化を行います")
            # 3次元配列の標準化の方法
            # https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix
            (x_train, x_val, x_test) = map(
                self.max_norm, (x_train, x_val, x_test)
            )
            # map(self.max_norm, (x_train, x_val, x_test))

        # 睡眠段階のラベルを0 -（クラス数-1）にする
        (y_train, y_val, y_test) = map(
            self.change_label,
            (y_train, y_val, y_test),
            (target_ss, target_ss, target_ss),
            (n_class_converted, n_class_converted, n_class_converted),
        )

        # inset_channel_axis based on data type
        if (
            self.data_type == "spectrum"
            or self.data_type == "cepstrum"
            and insert_channel_axis is True
        ):
            print("- チャンネル方向に軸を追加します")
            insert_channel = lambda x: x[:, :, np.newaxis]
            (x_train, x_val, x_test) = map(
                insert_channel, (x_train, x_val, x_test)
            )
        elif self.data_type == "spectrogram" and insert_channel_axis is True:
            print("- チャンネル方向に軸を追加します")
            insert_channel = lambda x: x[:, :, :, np.newaxis]
            (x_train, x_val, x_test) = map(
                insert_channel, (x_train, x_val, x_test)
            )

        # convert to one-hot vector
        if to_one_hot_vector:
            print("- one-hotベクトルを出力します")
            converter = lambda y: tf.one_hot(y, class_size)
            (y_train, y_val, y_test) = map(converter, (y_train, y_val, y_test))

        # only use under 4hz
        if is_under_4hz:
            print("- 4Hzで足切りをします")
            shaper_4hz = lambda x: x[:, : int(x.shape[1] / 2), :, :]
            (x_train, x_val, x_test) = map(
                shaper_4hz, (x_train, x_val, x_test)
            )

        if self.verbose == 0:
            print(
                "*** 全ての前処理後（one-hotを除く）の訓練データセット（確認用） *** \n",
                "y_train: ",
                f"{Counter(y_train)}\n",
                "y_val: ",
                f"{Counter(y_val)}\n",
                "y_test: ",
                f"{Counter(y_test)}\n",
            )
        return (
            (x_train, np.vstack([y_train, y_train_subject])),
            (x_val, np.vstack([y_val, y_val_subject])),
            (x_test, np.vstack([y_test, y_test_subject])),
        )

    # 文字列から睡眠段階の数字に変更するメソッド
    def ss2int(self, target_ss: list) -> list:
        __tmp = list()
        for _target_ss in target_ss:
            if _target_ss == "nr4":
                __tmp.append(0)
            elif _target_ss == "nr3":
                __tmp.append(1)
            elif _target_ss == "nr2":
                __tmp.append(2)
            elif _target_ss == "nr1":
                __tmp.append(3)
            elif _target_ss == "rem":
                __tmp.append(4)
            elif _target_ss == "wake":
                __tmp.append(5)
            else:
                print(
                    PyColor.RED_FLASH,
                    f"Unkown sleep stage name {_target_ss} given",
                    PyColor.END,
                )
                sys.exit(1)
        return __tmp

    # recordからスペクトラムの作成
    def list2Spectrum(self, list_data):
        return np.array([data.spectrum for data in list_data])

    # recordから睡眠段階の作成
    def list2SS(self, list_data):
        return np.array([k.ss for k in list_data])

    # recordから年齢の作成
    def list2age(self, list_data: List[Record]):
        return [k.age for k in list_data]

    # recordから被験者の作成
    def list2sub(self, list_data: List[Record]) -> ndarray:
        # 被験者名とそのラベルを定義
        for __record in list_data:
            if not __record.name in self.has_catched_name:
                self.has_catched_name.update(
                    {__record.name: self.subject_counter}
                )
                self.subject_counter += 1
        return np.array(
            [self.has_catched_name[__record.name] for __record in list_data]
        )

    # recordからスペクトログラムの作成
    def list2Spectrogram(self, list_data):
        return np.array([data.spectrogram for data in list_data])

    # make cepstrum from list
    def list2Cepstrum(self, list_data):
        return np.array([data.cepstrum for data in list_data])

    # 訓練データとテストデータをスプリット（ホールドアウト検証）
    def split_train_test_data(
        self, name=None, load_all=False, test_name=None, pse_data=False
    ):

        if pse_data:
            print("仮データです．データの読み込みは行いません")
            return None, None
        # 一人の被験者を入れたときはこの人だけ読み込む
        if name is not None:
            datas = self.load_sleep_data.load_data(name=name)
            return datas
        # 全員の被験者データを読み込む
        if load_all:
            datas = self.load_sleep_data.load_data(load_all=load_all)
        else:
            print("name, load_allのどちらかを指定してください")
            sys.exit(1)

        # NOTE : 一人以上の被験者を読み込んでいるときのみ以下実行可能

        # 被験者名を指定して訓練データとテストデータを分ける
        records_train = list()
        records_test = list()

        # NOTE : dataの順番と被験者の順番が等しいからできる処理
        for id, data in enumerate(datas):
            if test_name == self.name_list[id]:
                records_test.extend(data)
                print(f"testデータは{id}番目の{test_name}です")
            else:
                records_train.extend(data)

        return records_train, records_test

    # 訓練データとテストデータをスプリット（ホールドアウト検証（旧バージョン））
    def split_train_test_from_records(
        self, records, test_id, pse_data: bool = False
    ):
        # NOTE : pse_data is needed for avoiding split data
        if pse_data:
            print("仮データのためスキップします")
            return (None, None)
        test_records = records[test_id]
        train_records = list()
        for record_id, record in enumerate(records):
            if test_id != record_id:
                train_records.extend(record)
        return (train_records, test_records)

    # 訓練データのサイズをセットする
    def set_datasize(
        self,
        class_size: int,
        each_data_size: int,
        is_storchastic: bool,
    ):

        if is_storchastic:
            print(
                PyColor.GREEN_FLASH,
                "確率的サンプリング（2クラス分類時に他クラスのラベルを増やすために実装下部分です",
                PyColor.END,
            )
            return {
                1: each_data_size,
                2: each_data_size,
                3: each_data_size,
                4: each_data_size,
                5: each_data_size,
            }
        else:
            if class_size == 5:  # NR34, NR2, NR1, REM, WAKE
                if self.has_nrem2_bias:
                    if self.has_rem_bias:
                        return {
                            1: each_data_size,
                            2: each_data_size * 2,
                            3: each_data_size,
                            4: each_data_size * 2,
                            5: each_data_size,
                        }
                    else:
                        return {
                            1: each_data_size,
                            2: each_data_size * 2,
                            3: each_data_size,
                            4: each_data_size,
                            5: each_data_size,
                        }
                else:
                    if self.has_rem_bias:
                        return {
                            1: each_data_size,
                            2: each_data_size,
                            3: each_data_size,
                            4: each_data_size * 2,
                            5: each_data_size,
                        }

                    return {
                        1: each_data_size,
                        2: each_data_size,
                        3: each_data_size,
                        4: each_data_size,
                        5: each_data_size,
                    }
            elif class_size == 4:  # NR34, NR12, REM, WAKE
                return {
                    1: each_data_size,
                    2: each_data_size,
                    3: each_data_size,
                    4: each_data_size,
                }
            elif class_size == 3:  # NREM, REM, WAKE
                return {
                    1: each_data_size,
                    2: each_data_size,
                    3: each_data_size,
                }
            elif class_size == 2:  # TARGET, NON-TARGET
                return {1: each_data_size, 2: each_data_size}
            else:
                print("クラスサイズは5以下2以上に入れてください")
                sys.exit(1)

    # 各スペクトルの最大値で正規化
    def max_norm(self, data):
        data /= data.max()
        return data
        # for X in data:  # TODO : 全体の値で割るようなこともする
        #     X /= X.max()

    # 各スペクトルの最小値で正規化
    def min_norm(self, data):
        for X in data:  # TODO : 全体の値で割るようなこともする
            X /= X.min()

    # NONEの睡眠段階をキャッチして入力データごと消去
    def catch_none(self, x_data, y_data, y_data_subject):
        import pandas as pd

        x_data = list(x_data)
        y_data = list(y_data)
        y_data_subject = list(y_data_subject)
        # 保存用のリストを確保
        _x_data = list()
        _y_data = list()
        _y_data_subject = list()
        for num, ss in enumerate(y_data):
            if not pd.isnull(ss):
                _x_data.append(x_data[num])
                _y_data.append(y_data[num])
                _y_data_subject.append(y_data_subject[num])
            else:
                print("none data has catched")
        return (
            np.array(_x_data),
            np.array(_y_data).astype(np.int32),
            np.array(_y_data_subject).astype(np.int32),
        )

    # ラベルをクラス数に合わせて変更
    def change_label(
        self,
        y_data: list,
        target_class: list = [],
        n_class_converted: int = 5,
    ):
        nr4_label_from = 0
        nr3_label_from = 1
        nr2_label_from = 2
        nr1_label_from = 3
        rem_label_from = 4
        wake_label_from = 5
        # 5クラス分類の時（0:nr34, 1:nr2, 2:nr1, 3:rem, 4:wake）
        if n_class_converted == 5:
            nr4_label_to = 0
            nr3_label_to = 0
            nr2_label_to = 1
            nr1_label_to = 2
            rem_label_to = 3
            wake_label_to = 4
        # 4クラス分類の時（0:nr34, 1:nr12, 2:rem, 3:wake）
        elif n_class_converted == 4:
            nr4_label_to = 0
            nr3_label_to = 0
            nr2_label_to = 1
            nr1_label_to = 1
            rem_label_to = 2
            wake_label_to = 3
        # 3クラス分類の時（0:nrem, 1:rem, 2:wake）
        elif n_class_converted == 3:
            nr4_label_to = 0
            nr3_label_to = 0
            nr2_label_to = 0
            nr1_label_to = 0
            rem_label_to = 1
            wake_label_to = 2
        # 2クラス分類とき（0:non_target, 1:target）
        elif n_class_converted == 2:
            assert len(target_class) != 0
            nr4_label_to = 1 if "nr4" in target_class else 0
            nr3_label_to = 1 if "nr3" in target_class else 0
            nr2_label_to = 1 if "nr2" in target_class else 0
            nr1_label_to = 1 if "nr1" in target_class else 0
            rem_label_to = 1 if "rem" in target_class else 0
            wake_label_to = 1 if "wake" in target_class else 0
        else:
            print("正しいクラス数を指定してください")
            sys.exit(1)

        # ラベル変更
        for num, ss in enumerate(y_data):
            if ss == nr4_label_from:
                y_data[num] = nr4_label_to
            elif ss == nr3_label_from:
                y_data[num] = nr3_label_to
            elif ss == nr2_label_from:
                y_data[num] = nr2_label_to
            elif ss == nr1_label_from:
                y_data[num] = nr1_label_to
            elif ss == rem_label_from:
                y_data[num] = rem_label_to
            elif ss == wake_label_from:
                y_data[num] = wake_label_to
        return y_data

    # データ拡張の一つ
    def smote(self, x, y, sample_num):
        sm = SMOTE(random_state=42, k_neighbors=4)
        x, y = sm.fit_resample(x[:sample_num], y[:sample_num])
        return x, y

    # tutorialにありそうなデータ拡張
    def dataArg(
        self,
        featurewise_center=True,
        featurewise_std_nomalization=True,
        # rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False,
    ):
        datagen = ImageDataGenerator(
            featurewise_center=featurewise_center,
            featurewise_std_normalization=featurewise_std_nomalization,
            # rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
        )
        return datagen

    def make_generator(self):
        train_image_generator = ImageDataGenerator(
            rescale=1.0 / 255,
            horizontal_flip=True,
            rotation_range=45,
            zoom_range=0.5,
        )
        validation_image_generator = ImageDataGenerator(rescale=1.0 / 255)
        train_data_gen = train_image_generator.flow_from_directory(
            batch_size=8,
            directory=self.train_dir,
            shuffle=True,
            target_size=(224, 224),
            class_mode="binary",
        )
        val_data_gen = validation_image_generator.flow_from_directory(
            batch_size=8,
            directory=self.val_dir,
            target_size=(224, 224),
            class_mode="binary",
        )
        test_data_gen = validation_image_generator.flow_from_directory(
            batch_size=8,
            directory=self.test_dir,
            target_size=(224, 224),
            class_mode="binary",
        )
        return train_data_gen, val_data_gen, test_data_gen

    def makePreProcessedMnist(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        return (x_train, y_train), (x_test, y_test)

    # 仮データの作成
    def make_pse_sleep_data(
        self,
        n_class: int,
        to_one_hot_vector: bool,
        data_size=1000,
        min_val=-1,
        max_val=1,
    ):
        if self.data_type == "spectrum":
            print("spectrum型の仮データを作成します")
            input_shape = (data_size, int(self.kernel_size / 2), 1)
        elif self.data_type == "spectrogram":
            print("spectrogram型の仮データを作成します")
            input_shape = (data_size, 128, 512, 1)
        else:
            raise Exception("unknown data type")
        # x_train = tf.random.normal(shape=input_shape)
        x_train = tf.random.uniform(
            shape=input_shape, minval=min_val, maxval=max_val
        )
        # x_test = tf.random.normal(shape=input_shape)
        x_test = tf.random.uniform(
            shape=input_shape, minval=min_val, maxval=max_val
        )
        y_train = np.random.randint(0, n_class - 1, size=data_size)
        y_test = np.random.randint(0, n_class - 1, size=data_size)
        if to_one_hot_vector:
            y_train = np.array(tf.one_hot(y_train, depth=n_class))
            y_test = np.array(tf.one_hot(y_train, depth=n_class))
        return (x_train, y_train), (x_test, y_test)

    # 平行移動を行う前処理
    def random_shift(self, x, wrg, hrg):
        try:
            assert type(x) == np.ndarray
        except TypeError:
            print("xはnumpyの形式で渡してください")
        shifted_x = tf_image.random_shift(x, wrg=wrg, hrg=hrg)
        return shifted_x


if __name__ == "__main__":
    import rich

    from data_analysis.utils import Utils

    PSE_DATA = False
    utils = Utils(
        is_normal=True,
        is_previous=False,
        data_type="spectrogram",
        fit_pos="middle",
        stride=128,
        kernel_size=16,
        model_type="enn",
        cleansing_type="nothing",
        catch_nrem2=True,
    )
    pre_process = PreProcess(
        data_type="spectrogram",
        fit_pos="middle",
        verbose=0,
        kernel_size=128,
        is_previous=False,
        stride=16,
        is_normal=True,
        cleansing_type="nothing",
        has_nrem2_bias=True,
        model_type="enn",
    )
    # 全員読み込む
    records = pre_process.load_sleep_data.load_data(
        load_all=True, pse_data=PSE_DATA, name=None
    )
    # テストと訓練にスプリット
    (train, test) = pre_process.split_train_test_from_records(
        records=records, test_id=0, pse_data=PSE_DATA
    )
    # 読み込みとsplitを同時に行う
    # (train, test) = pre_process.split_train_test_data(
    #     load_all=True, test_name="H_Li")
    (x_train, y_train), (x_test, y_test) = pre_process.make_dataset(
        train=train,
        test=test,
        is_storchastic=False,
        pse_data=PSE_DATA,
        to_one_hot_vector=False,
        each_data_size=100,
        normalize=False,
    )
    ss = 5
    num_plot = 16
    print(Counter(y_test[0]))
    utils.plotly_images(
        images_arr=x_test,
        title_arr=y_test[0],
        num_plot=num_plot,
    )
    # print("x_train.shape: ", x_train.shape)
    # print("y_train.shape: ", y_train.shape)
