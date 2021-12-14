import datetime
import os
import sys
from collections import Counter
from random import choices, seed, shuffle

import numpy as np
import tensorflow as tf
import tensorflow.keras.preprocessing.image as tf_image
from imblearn.over_sampling import SMOTE
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_analysis.py_color import PyColor
from pre_process.file_reader import FileReader
from pre_process.load_sleep_data import LoadSleepData
from pre_process.my_env import MyEnv

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
    ):
        seed(0)
        self.date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.data_type = data_type
        self.fit_pos = fit_pos
        self.verbose = verbose
        self.kernel_size = kernel_size
        self.is_previous = is_previous
        self.stride = stride
        self.model_type = model_type
        self.is_normal = is_normal
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
        )
        self.has_nrem2_bias = has_nrem2_bias
        self.has_rem_bias = has_rem_bias
        # LoadSleepData の参照を作成
        self.fr: FileReader = self.load_sleep_data.fr
        self.sl = self.load_sleep_data.sl
        self.my_env: MyEnv = self.load_sleep_data.my_env
        # その他よく使うものをメンバに持っておく
        self.name_list = self.sl.set_name_list(
            is_previous=self.is_previous, is_normal=self.is_normal
        )

    # データセットの作成（この中で出来るだけ正規化なども含めて終わらせる）
    # TODO: データ選択方法の見直し
    def make_dataset(
        self,
        train=None,
        test=None,
        is_set_data_size=True,
        target_ss=None,
        is_storchastic=True,
        is_shuffle=True,
        is_multiply=False,
        mul_num=None,
        each_data_size=1000,
        class_size=5,
        normalize=True,
        catch_none=True,
        insert_channel_axis=True,
        to_one_hot_vector=True,
        pse_data=False,
    ):
        # NOTE : when true, make pse_data based on the data type
        # which specified in load_sleep_data object
        if pse_data:
            print("- 仮の睡眠データを作成します")
            return self.make_pse_sleep_data(
                n_class=class_size,
                data_size=each_data_size,
                to_one_hot_vector=to_one_hot_vector,
            )

        if is_set_data_size:

            if self.verbose == 0:
                print("- 訓練データのサイズを揃えます")

            # 各睡眠段階のサイズを決定する
            ss_dict_train = self.set_datasize(
                train,
                target_ss=target_ss,
                isStorchastic=is_storchastic,
                each_data_size=each_data_size,
                class_size=class_size,
            )
            ss_dict_test = Counter([record.ss for record in test])

            if self.verbose == 0:
                print(
                    "訓練データの各睡眠段階（補正前）",
                    Counter([record.ss for record in train]),
                )
                print("訓練データの各睡眠段階（補正後）", ss_dict_train)

            def _storchastic_sampling(data, target_records, is_test):
                tmp = list()

                def _splitEachSleepStage():
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

                # record を各睡眠段階ごとにわけているもの（長さ５とは限らない（nr3がない場合））
                ss_list = _splitEachSleepStage()
                # 各睡眠段階ごとに処理を行う
                for records in ss_list:
                    if len(records) == 0:
                        print("睡眠段階が存在しないため飛ばします")
                        continue
                    # 各睡眠段階がある間はその睡眠段階に対してランダムサンプリングを行う
                    else:

                        def _sample(target_records):
                            # 睡眠段階のラベルを知るために必要
                            if is_test:
                                # ss = target_records[0].ss
                                # _selected_list = choices(
                                #     target_records, k=data[ss]
                                # )
                                print(
                                    PyColor.RED_FLASH,
                                    f"廃止しました。テスト時は{sys._getframe().f_code.co_name}を使用しないようにしてください",
                                    PyColor.END,
                                )
                            else:
                                ss = target_records[0].ss
                                if ss != target_ss:
                                    try:
                                        # target_recordsからdata[ss]個取ってくる感じ
                                        # >>> len(target_records)
                                        # >>> 1019（各睡眠段階のrecordオブジェクトが入っている
                                        # >>> data[4]
                                        # >>> 睡眠段階が4のもの（REM）の数を返す
                                        # （正確にはCounterの辞書の4に対応するvalueのこと）
                                        if is_multiply:
                                            _selected_list = choices(
                                                target_records, k=400
                                            )
                                        else:
                                            if mul_num is not None:
                                                _selected_list = choices(
                                                    target_records,
                                                    k=data[ss] * mul_num,
                                                )
                                            else:
                                                _selected_list = choices(
                                                    target_records, k=data[ss]
                                                )
                                    except Exception:
                                        if is_multiply:
                                            # print("データを複製する必要があるので、random.choices()を用います")
                                            _selected_list = choices(
                                                target_records, k=400
                                            )
                                        else:
                                            if mul_num is not None:
                                                _selected_list = choices(
                                                    target_records,
                                                    k=data[ss] * mul_num,
                                                )
                                            else:
                                                _selected_list = choices(
                                                    target_records, k=data[ss]
                                                )
                                elif ss == target_ss:
                                    _selected_list = choices(
                                        target_records, k=data[ss]
                                    )
                            return _selected_list

                    tmp.extend(_sample(target_records=records))
                return tmp

            train = _storchastic_sampling(
                data=ss_dict_train, target_records=train, is_test=False
            )
            # test = _storchastic_sampling(
            #     data=ss_dict_test, target_records=test, is_test=True
            # )

            if is_shuffle:
                print("- 訓練データをシャッフルします")
                shuffle(train)

        else:
            if self.verbose == 0:
                print("データサイズをそろえずにデータセットを作成します")

        # TODO : スペクトログラムかスペクトラム化によって呼び出す関数を場合分け
        y_train = self.list2SS(train)
        y_test = self.list2SS(test)
        if self.data_type == "spectrum":
            x_train = self.list2Spectrum(train)
            x_test = self.list2Spectrum(test)
        elif self.data_type == "spectrogram":
            x_train = self.list2Spectrogram(train)
            x_test = self.list2Spectrogram(test)
        elif self.data_type == "cepstrum":
            x_train = self.list2Cepstrum(train)
            x_test = self.list2Cepstrum(test)
        else:
            print("spectrum or spectrogram or cepstrumを指定してください")
            sys.exit(1)

        # Noneの処理をするかどうか
        if catch_none is True:
            print("- noneの処理を行います")
            x_train, y_train = self.catch_none(x_train, y_train)
            x_test, y_test = self.catch_none(x_test, y_test)

        # max正規化をするかどうか
        if normalize is True:
            print("- max正規化を行います")
            self.max_norm(x_train)
            self.max_norm(x_test)
            # self.min_norm(x_train)
            # self.min_norm(x_test)

        # 睡眠段階のラベルを0 -（クラス数-1）にする
        y_train = self.change_label(
            y_data=y_train, n_class=class_size, target_class=target_ss
        )
        y_test = self.change_label(
            y_data=y_test, n_class=class_size, target_class=target_ss
        )

        # inset_channel_axis based on data type
        if self.data_type == "spectrum" or self.data_type == "cepstrum":
            if insert_channel_axis:
                print("- チャンネル方向に軸を追加します")
                x_train = x_train[:, :, np.newaxis]  # .astype('float32')
                x_test = x_test[:, :, np.newaxis]  # .astype('float32')
        elif self.data_type == "spectrogram":
            if insert_channel_axis:
                print("- チャンネル方向に軸を追加します")
                x_train = x_train[:, :, :, np.newaxis]  # .astype('float32')
                x_test = x_test[:, :, :, np.newaxis]  # .astype('float32')

        if self.verbose == 0:
            print(
                "*** 全ての前処理後（one-hotを除く）の訓練データセット（確認用） *** \n",
                Counter(y_train),
            )

        # convert to one-hot vector
        if to_one_hot_vector:
            print("- one-hotベクトルを出力します")
            y_train = tf.one_hot(y_train, class_size)
            y_test = tf.one_hot(y_test, class_size)

        return (x_train, y_train), (x_test, y_test)

    # recordからスペクトラムの作成
    def list2Spectrum(self, list_data):
        return np.array([data.spectrum for data in list_data])

    # recordから睡眠段階の作成
    def list2SS(self, list_data):
        return np.array([k.ss for k in list_data])

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
        ss_list,
        target_ss,
        isStorchastic,
        each_data_size,
        class_size,
    ):

        if isStorchastic:
            print("確率的サンプリングを実装し直しましょう")
            sys.exit(1)
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
        for X in data:  # TODO : 全体の値で割るようなこともする
            X /= X.max()

    # 各スペクトルの最小値で正規化
    def min_norm(self, data):
        for X in data:  # TODO : 全体の値で割るようなこともする
            X /= X.min()

    # NONEの睡眠段階をキャッチして入力データごと消去
    def catch_none(self, x_data, y_data):
        import pandas as pd

        x_data = list(x_data)
        y_data = list(y_data)
        # 保存用のリストを確保
        _x_data = list()
        _y_data = list()
        for num, ss in enumerate(y_data):
            if not pd.isnull(ss):
                _x_data.append(x_data[num])
                _y_data.append(y_data[num])
        return np.array(_x_data), np.array(_y_data).astype(np.int32)

    # ラベルをクラス数に合わせて変更
    def change_label(self, y_data, n_class, target_class=None):
        nr4_label_from = 0
        nr3_label_from = 1
        nr2_label_from = 2
        nr1_label_from = 3
        rem_label_from = 4
        wake_label_from = 5
        # 5クラス分類の時（0:nr34, 1:nr2, 2:nr1, 3:rem, 4:wake）
        if n_class == 5:
            nr4_label_to = 0
            nr3_label_to = 0
            nr2_label_to = 1
            nr1_label_to = 2
            rem_label_to = 3
            wake_label_to = 4
        # 4クラス分類の時（0:nr34, 1:nr12, 2:rem, 3:wake）
        elif n_class == 4:
            nr4_label_to = 0
            nr3_label_to = 0
            nr2_label_to = 1
            nr1_label_to = 1
            rem_label_to = 2
            wake_label_to = 3
        # 3クラス分類の時（0:nrem, 1:rem, 2:wake）
        elif n_class == 3:
            nr4_label_to = 0
            nr3_label_to = 0
            nr2_label_to = 0
            nr1_label_to = 0
            rem_label_to = 1
            wake_label_to = 2
        # 2クラス分類とき（0:non_target, 1:target）
        elif n_class == 2:
            assert target_class is not None
            nr4_label_to = 0 if target_class != "nr4" else 1
            nr3_label_to = 0 if target_class != "nr3" else 1
            nr2_label_to = 0 if target_class != "nr2" else 1
            nr1_label_to = 0 if target_class != "nr1" else 1
            rem_label_to = 0 if target_class != "rem" else 1
            wake_label_to = 0 if target_class != "wake" else 1
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
    from pre_process.load_sleep_data import LoadSleepData

    PSE_DATA = False
    load_sleep_data = LoadSleepData(data_type="spectrum", verbose=0, n_class=5)
    pre_process = PreProcess(load_sleep_data)
    # 全員読み込む
    records = load_sleep_data.load_data(
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
        train=train, test=test, is_storchastic=False, pse_data=PSE_DATA
    )
    print(x_train.shape)
    # 1d が本当にデータ入っているかの確認 -> 確認済み
    # import matplotlib.pyplot as plt
    # plt.plot(x_train[0])
    # plt.show()
