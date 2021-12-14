import datetime
import sys
from collections import Counter
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from scipy.signal import hamming

from data_analysis.py_color import PyColor
from pre_process.record import Record, make_mul_records


class CreateData(object):
    def __init__(self):
        return

    # 窓関数のどの位置の睡眠段階を決定するかのメソッド
    def _set_fit_pos_function(self, fit_pos: str) -> Callable:
        def __set_top(start_point: int, end_point: int):
            return start_point

        def __set_middle(start_point: int, end_point: int):
            return int((start_point + end_point) / 2)

        def __set_bottom(start_point: int, end_point: int):
            return end_point

        if fit_pos == "top":
            return __set_top
        elif fit_pos == "middle":
            return __set_middle
        elif fit_pos == "bottom":
            return __set_bottom
        else:
            print("exception occured")
            sys.exit(1)

    # 実際のスペクトル作成と時間の記録
    def _make(
        self,
        mode: str,
        start_point: int,
        record: Record,
        kernel_size: int,
        tanita_data: DataFrame,
        psg_data: DataFrame,
        set_fit_pos_func: Callable,
        time_psg: datetime,
        psg_len: int,
        ss_term: int = 0,
        stride: int = 0,
    ) -> tuple:
        if mode == "spectrum":
            end_point = start_point + kernel_size - 1
            try:
                amped = (
                    hamming(
                        len(tanita_data["ss"][start_point : end_point + 1])
                    )
                    * tanita_data["ss"][start_point : end_point + 1]
                )
            except IndexError:
                print(f"{end_point+1}は tanita の長さを超えてます")
            fft = np.fft.fft(amped) / (kernel_size / 2.0)
            fft = 20 * np.log10(np.abs(fft))
            fft = fft[: int(kernel_size / 2)]
            record.spectrum = fft
            fit_index = set_fit_pos_func(start_point, end_point)
            # NOTE: なぜ .iloc を使う必要があるのか
            record.time = tanita_data["time"].iloc[fit_index]
            _time_tanita = datetime.datetime.strptime(
                tanita_data["time"].iloc[end_point], "%H:%M:%S"
            )
            _time_delta = time_psg - _time_tanita
            # 時刻がオーバー（負）、かつ経過秒数が22時間以上であれば終了したとみなす
            # （日付がデータに入っていればもっと正確にできる） tanitaのけつの時間と比較する
            if _time_delta.days < 0 and _time_delta.seconds / 60 / 60 > 22:
                return False, None
            # psgのデータサイズよりも大きいインデックスを指定していれば終了する（上でキャッチできない場合）
            elif psg_len <= fit_index // 16:
                print(PyColor.RED_FLASH, "実装上あまりここには来てほしくない", PyColor.END)
                return False, None
            else:
                return True, fit_index

        elif mode == "cepstrum":
            end_point = start_point + kernel_size - 1
            try:
                amped = (
                    hamming(
                        len(tanita_data["ss"][start_point : end_point + 1])
                    )
                    * tanita_data["ss"][start_point : end_point + 1]
                )
            except IndexError:
                print(f"{end_point+1}は tanita の長さを超えてます")
            # ケプストラムの計算
            fft = np.fft.fft(amped) / (kernel_size / 2.0)
            fft = 20 * np.log10(np.abs(fft))
            # fft = fft[: int(kernel_size / 2)]
            ifft = np.fft.ifft(fft)
            ifft = np.real(ifft)
            # ifft = np.real(np.fft.ifft(fft)) / (kernel_size / 2.0)
            ifft = 20 * np.log10(np.abs(ifft))

            # スペクトル包絡にする
            # fft = 20 * np.log10(np.abs(np.fft.fft(amped)))
            # cps = np.real(np.fft.ifft(fft))
            # cep_coef = 20
            # # 高周波成分を除く
            # cps_lif = np.array(cps)
            # cps_lif[cep_coef : len(cps_lif) - cep_coef + 1] = 0
            # ifft = np.real(np.fft.fft(cps_lif))
            # 差分を取る
            # _ifft = np.append(ifft[1:], ifft[-1])
            # ifft -= _ifft
            record.cepstrum = ifft[: int(kernel_size / 2)]
            fit_index = set_fit_pos_func(start_point, end_point)
            # NOTE: なぜ .iloc を使う必要があるのか
            record.time = tanita_data["time"].iloc[fit_index]
            _time_tanita = datetime.datetime.strptime(
                tanita_data["time"].iloc[end_point], "%H:%M:%S"
            )
            _time_delta = time_psg - _time_tanita
            # 時刻がオーバー（負）、かつ経過秒数が22時間以上であれば終了したとみなす
            # （日付がデータに入っていればもっと正確にできる） tanitaのけつの時間と比較する
            if _time_delta.days < 0 and _time_delta.seconds / 60 / 60 > 22:
                return False, None
            # psgのデータサイズよりも大きいインデックスを指定していれば終了する（上でキャッチできない場合）
            elif psg_len <= fit_index // 16:
                print(PyColor.RED_FLASH, "実装上あまりここには来てほしくない", PyColor.END)
                return False, None
            else:
                return True, fit_index

        elif mode == "spectrogram":
            end_point = start_point + kernel_size * ss_term - 1
            fit_index = set_fit_pos_func(start_point, end_point)
            # 後に周波数変換後のデータを格納用のダミー配列
            spectrogram = np.array(
                [i for i in range(kernel_size // 2)]
            ).reshape(kernel_size // 2, -1)
            # TODO: 何に基づいて下のループを回しているのか？
            # スペクトログラム内で窓をずらしながらスペクトル作成(30回)＋結合
            for _ss_term in range(ss_term):
                # 0814_shiraishi のデータでtanita_data["ss"][...]のオブジェクトタイプがfloatと読み込まれず掛け算が出来ないので、numpyに変換したら大丈夫か検証中
                try:
                    amped = hamming(
                        len(
                            tanita_data["ss"][
                                start_point
                                + _ss_term * 16 : start_point
                                + _ss_term * 16
                                + kernel_size  # 16は_ss_termの単位が秒なのでタニタのデータに関して16Hzなので16倍進める
                            ]
                        )
                    ) * tanita_data["ss"][
                        start_point
                        + _ss_term * 16 : start_point
                        + _ss_term * 16
                        + kernel_size  # 16は_ss_termの単位が秒なのでタニタのデータに関して16Hzなので16倍進める
                    ].astype(
                        np.float16
                    )
                except IndexError:
                    print(f"{end_point+1}は tanita の長さを超えてます")
                fft = np.fft.fft(amped) / (kernel_size / 2.0)
                fft = 20 * np.log10(np.abs(fft))
                fft = fft[: int(kernel_size / 2)]
                spectrogram = np.hstack(
                    [spectrogram, fft.reshape(kernel_size // 2, 1)]
                )
            # 1列目はappendが出来るように入れていたダミーデータのため保存の前に除去（obj:0の要素，axis:1（列方向））
            spectrogram = np.delete(spectrogram, obj=0, axis=1)
            record.spectrogram = spectrogram
            record.time = tanita_data["time"].iloc[fit_index]
            # NOTE: 付け焼刃なので後で吟味
            # もしtanitaの長さ以上を超えていたらFalseを返す
            # tanitaが終了時刻まで取れてないのであれば終了
            if len(tanita_data) <= end_point:
                print("tanita_len", len(tanita_data), "end_point", end_point)
                return False, None
            # tanitaは取れてるけど、psgが取れてない時の終了条件が下
            _time_tanita = datetime.datetime.strptime(
                tanita_data["time"].iloc[end_point], "%H:%M:%S"
            )
            # PSGの最後の時刻とFFT中のtanitaのデータを比較する
            _time_delta = time_psg - _time_tanita
            # _time_delta = _time_tanita - time_psg
            # 以下2条件を同時に満たす場合はtanitaのデータがpsgのデータの最後の時刻を超えているため終了とする条件である
            # 1. psgデータの方がtanitaデータの方よりも時刻が低い（min:0時00分00秒，max:23時59分59秒．）NOTE:日付があればこんな処理をする必要はないが、各データに入れるのがめんどくさいためこの方法で代替
            # 2. 22時間以上離れている
            # （ex. 0時をまたぐ場合にtanita: 23:59:59, psg: 00:00:00の場合、睡眠は続くので以降も前処理を続けるべきなので単純に_time_delta.days < 0 だけでは判断できない）
            # そのためそのようなケースは22時間以上
            # タニタとPSGの差が1時間以内 かつ経過方向が正（タニタの方が後の時刻）であれば終了とする
            # if _time_delta.seconds / 60 / 60 < 1 and _time_delta.days == 0:
            #     return False, None
            if _time_delta.days < 0 and _time_delta.seconds / 60 / 60 > 22:
                return False, None
            # tanitaの前処理の最終時刻がpsgの最終時刻より前でも、psgのデータサイズよりも大きいインデックスを指定していれば終了する（上でキャッチできない場合）
            elif psg_len <= fit_index // 16:
                print(PyColor.RED_FLASH, "実装上あまりここには来てほしくない", PyColor.END)
                return False, None
            else:
                return True, fit_index
        else:
            print("実装はまだされていません")
            sys.exit(1)

    def _match(
        self,
        record: Record,
        start_point: int,
        fit_index: int,
        psg_data: DataFrame,
    ) -> None:
        # psgの時間とrecordの時間(tanita)が等しい時刻のときの睡眠段階を代入
        # まず公式に合致すれば，ループ処理をしなくて済む
        # NOTE : 16Hzはタニタのセンサ固有の値
        # トリムしてある場合はループ処理をしないので超高速
        # record.timeによって終了条件確認
        # 1. Noneであれば、例外としてcontinueする
        if record.time is None:
            print(PyColor.RED_FLASH, "Noneの時刻が入ってきました", PyColor.END)
        try:
            record.ss = psg_data["ss"][fit_index]
        except IndentationError:
            print(
                PyColor.RED_FLASH,
                f"{fit_index}はpsgの長さを超えています",
                PyColor.END,
            )
            sys.exit(1)
        # 睡眠段階がNoneであればストップ
        if record.ss is None:
            print("stop")

    # スペクトルの作成などのメタメソッド
    def make_freq_transform(
        self,
        mode: str,
    ) -> Callable:
        if mode == "spectrum":
            return self.make_spectrum
        elif mode == "spectrogram":
            return self.make_spectrogram
        elif mode == "cepstrum":
            return self.make_cepstrum
        else:
            print("実装まだです")
            sys.exit(1)

    # ケプストラムの作成
    def make_cepstrum(
        self,
        tanita_data: DataFrame,
        psg_data: DataFrame,
        kernel_size: int,
        stride: int,
        fit_pos: str = "middle",
    ) -> list:
        # TODO: タニタのデータが22時間以上の長さがないことを確認（tanitaの初期化のときすればいいかも）
        # 一般的に畳み込みの回数は (data_len - kernel_len) / stride + 1
        record_len = int((len(tanita_data) - kernel_size) / stride) + 1
        print(f"make {record_len} spectrum datas")
        records = make_mul_records(record_len)
        # スタートポイントはストライドのサイズでずらして、
        # データを作成する回数分のリストを確保する
        start_points_list = [i for i in range(0, stride * record_len, stride)]
        # psgのデータ数を保持
        psg_len, _ = psg_data.shape
        # psgの最後の時間を保存しておく（_match用）
        _time_psg = datetime.datetime.strptime(
            psg_data["time"].iloc[-1], "%H:%M:%S"
        )

        # 窓関数のどの位置を睡眠段階としてとるか決める関数を設定
        _set_fit_pos_func = self._set_fit_pos_function(fit_pos=fit_pos)

        for start_point, record in zip(start_points_list, records):
            (can_make, _fit_index) = self._make(
                mode="cepstrum",
                start_point=start_point,
                record=record,
                kernel_size=kernel_size,
                tanita_data=tanita_data,
                psg_data=psg_data,
                set_fit_pos_func=_set_fit_pos_func,
                time_psg=_time_psg,
                psg_len=psg_len,
            )
            if not can_make:
                break
            self._match(
                start_point=start_point,
                record=record,
                fit_index=int(_fit_index / 16),
                psg_data=psg_data,
            )

        data = [record.ss for record in records]
        print(f"睡眠段階：{Counter(data)}")

        return records

    # スペクトルの作成
    def make_spectrum(
        self,
        tanita_data: DataFrame,
        psg_data: DataFrame,
        kernel_size: int,
        stride: int,
        fit_pos: str = "middle",
    ) -> list:
        # TODO: タニタのデータが22時間以上の長さがないことを確認（tanitaの初期化のときすればいいかも）
        # 一般的に畳み込みの回数は (data_len - kernel_len) / stride + 1
        record_len = int((len(tanita_data) - kernel_size) / stride) + 1
        print(f"make {record_len} spectrum datas")
        records = make_mul_records(record_len)
        # スタートポイントはストライドのサイズでずらして、
        # データを作成する回数分のリストを確保する
        start_points_list = [i for i in range(0, stride * record_len, stride)]
        # psgのデータ数を保持
        psg_len, _ = psg_data.shape
        # psgの最後の時間を保存しておく（_match用）
        _time_psg = datetime.datetime.strptime(
            psg_data["time"].iloc[-1], "%H:%M:%S"
        )

        # 窓関数のどの位置を睡眠段階としてとるか決める関数を設定
        _set_fit_pos_func = self._set_fit_pos_function(fit_pos=fit_pos)

        for start_point, record in zip(start_points_list, records):
            (can_make, _fit_index) = self._make(
                mode="spectrum",
                start_point=start_point,
                record=record,
                kernel_size=kernel_size,
                tanita_data=tanita_data,
                psg_data=psg_data,
                set_fit_pos_func=_set_fit_pos_func,
                time_psg=_time_psg,
                psg_len=psg_len,
            )
            if not can_make:
                break
            self._match(
                start_point=start_point,
                record=record,
                fit_index=int(_fit_index / 16),
                psg_data=psg_data,
            )

        data = [record.ss for record in records]
        print(f"睡眠段階：{Counter(data)}")

        return records

    def make_spectrogram(
        self,
        tanita_data: DataFrame,
        psg_data: DataFrame,
        stride: int,
        fit_pos: str,
        kernel_size: int,
        ss_term: int = 30,
    ):
        # tanitaのデータからスペクトログラムの作成時のインデントを取得
        # NOTE: スペクトログラムに関して畳み込みができる回数をしっかり調べる必要がある
        # recordオブジェクトの必要な数．タニタのデータから算出
        record_len = (
            int((len(tanita_data) - kernel_size) / ss_term / stride) + 1
        )
        records = make_mul_records(record_len)
        # stride * ss_term
        start_points_list = [
            i
            for i in range(0, stride * record_len * ss_term, stride * ss_term)
        ]
        # psgのデータ数を取得
        psg_len, _ = psg_data.shape
        # psgの最後の時間を保存しておく（_match用）
        _time_psg = datetime.datetime.strptime(
            psg_data["time"].iloc[-1], "%H:%M:%S"
        )
        # 窓の関数のどの時刻の睡眠段階を正解ラベルとするか
        _set_fit_pos_func = self._set_fit_pos_function(fit_pos=fit_pos)

        # スペクトログラムの作成
        try:
            assert len(start_points_list) == len(records)
        except AssertionError as AE:
            print(AE)
        for start_point, record in zip(start_points_list, records):
            can_make, _fit_index = self._make(
                mode="spectrogram",
                start_point=start_point,
                record=record,
                kernel_size=kernel_size,
                tanita_data=tanita_data,
                psg_data=psg_data,
                set_fit_pos_func=_set_fit_pos_func,
                time_psg=_time_psg,
                psg_len=psg_len,
                ss_term=ss_term,
                stride=stride,
            )
            # スペクトログラムを作成できなかったら終了
            if not can_make:
                break
            self._match(
                start_point=start_point,
                record=record,
                fit_index=int(_fit_index / 16),
                psg_data=psg_data,
            )
        data = [record.ss for record in records]
        print(PyColor().YELLOW, f"睡眠段階:{Counter(data)}", PyColor().END)
        return records

    def makeCepstrogram(
        self, tanita_data, psg_data, sampleLen=1024, timeLen=128
    ):
        loopLen = int((len(tanita_data) - sampleLen) / 4)  # fft ができる回数
        recordLen = int(
            loopLen / timeLen
        )  # record オブジェクトを作る回数（128回のfftに関して1回作る）
        records = make_mul_records(recordLen)

        def _make():
            for i, record in enumerate(records):
                spectroGram = list()
                stop = 0
                for k in range(timeLen):
                    start = k * 4 + (4 * timeLen) * i
                    stop = start + sampleLen
                    if stop > len(tanita_data):
                        print(
                            f"stop index is {stop}.",
                            "this is out of range tanita",
                            f"data {len(tanita_data)}",
                        )
                        break
                    # FFT 変換（スペクトラム）
                    amped = (
                        hamming(len(tanita_data["val"][start:stop]))
                        * tanita_data["val"][start:stop]
                    )
                    fft = np.fft.fft(amped) / (
                        len(tanita_data["val"][start:stop]) / 2.0
                    )
                    fft = np.log10(np.abs(fft))
                    # FFT 変換（ケプストラム）
                    fft = np.fft.fft(fft) / (len(fft) / 2.0)
                    fft = 20 * np.log10(np.abs(fft))
                    # リスト変換
                    fft = list(
                        fft[: int(len(tanita_data["val"][start:stop]) / 2.0)]
                    )
                    spectroGram.append(fft)
                record.spectrogram = spectroGram
                if stop > len(tanita_data):
                    break
                record.time = tanita_data["time"][stop]

                def _match():
                    for it, time in enumerate(psg_data["time"]):
                        if time == record.time:
                            return psg_data["ss"][it]

                record.ss = _match()
                if record.ss is None:
                    break
            return records

        return _make()


if __name__ == "__main__":
    from pre_process.psg_reader import PsgReader
    from pre_process.tanita_reader import TanitaReader

    CD = CreateData()
    tanita = TanitaReader("H_Hayashi")
    tanita.readCsv()
    psg = PsgReader("H_Hayashi")
    psg.readCsv()
    record = Record()
    data = tanita.df["val"][0 : 1024 + 511]
    spectroGram = CD.makeSpectrogram(tanita_data=tanita.df, psg_data=psg.df)
    plt.imshow(record.spectrogram)
