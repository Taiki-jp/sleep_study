from data_analysis.py_color import PyColor
from pre_process.record import Record, multipleRecords
import numpy as np
from scipy.signal import hamming
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys


class CreateData(object):
    def __init__(self):
        return

    def makeSpectrum(
        self, tanita_data, psg_data, kernel_size, stride, fit_pos="middle"
    ):
        # 一般的に畳み込みの回数は (data_len - kernel_len) / stride + 1
        # ↑図を描くとわかりやすい
        record_len = int((len(tanita_data) - kernel_size) / stride) + 1
        print(f"make {record_len} datas")
        records = multipleRecords(record_len)
        start_points_list = [i for i in range(0, stride * record_len, stride)]
        # TODO : recordを無駄に作成してしまう恐れあり（本当はend_pointに合わせてレコードを作ったほうが良い？）
        try:
            assert record_len == len(start_points_list)
        except AssertionError:
            print(
                "tanita_len",
                len(tanita_data),
                "record len",
                record_len,
                "start point len",
                len(start_points_list),
            )
            sys.exit(3)

        def _make(start_point, record):
            end_point = start_point + kernel_size - 1
            try:
                amped = (
                    hamming(
                        len(
                            tanita_data["sensor1"][start_point : end_point + 1]
                        )
                    )
                    * tanita_data["sensor1"][start_point : end_point + 1]
                )
            except Exception:
                print("tmp stop")
            fft = np.fft.fft(amped) / (
                len(tanita_data["sensor1"][start_point : end_point + 1]) / 2.0
            )
            fft = 20 * np.log10(np.abs(fft))
            fft = fft[: int(kernel_size / 2)]
            record.spectrum = fft
            if fit_pos == "top":
                fit_index = start_point
            elif fit_pos == "middle":
                fit_index = int((start_point + end_point) / 2)
            elif fit_pos == "bottom":
                fit_index = end_point
            else:
                print("exception occured")
                sys.exit(1)
            record.time = tanita_data["time"].iloc[fit_index]

        def _match(record, start_point):
            # psgの時間とrecordの時間(tanita)が等しい時刻のときの睡眠段階を代入
            # まず公式に合致すれば，ループ処理をしなくて済む
            # NOTE : 16Hzはタニタのセンサ固有の値
            if fit_pos == "top":
                _index = int(start_point / 16)
            elif fit_pos == "middle":
                _index = int((start_point + (kernel_size / 2)) / 16)
            elif fit_pos == "bottom":
                _index = int((start_point + kernel_size) / 16) - 1
            else:
                print("exception occured")
                sys.exit(1)

            # インデックスがpsgのサイズを超えていないかどうか
            # タニタのセンサのけつがそろってないと起こりうる
            # これを回避したければ，タニタのけつの時間をpsgにそろえる
            if (_index + 1) >= psg_data.shape[0]:
                print(
                    PyColor.RED, "tanita data is out of psg data", PyColor.END
                )
                return False

            else:
                # 公式に合致する際はループを回さなくて済む
                # datetime オブジェクトを使用する
                # if datetime.datetime.strptime(psg_data["time"].iloc[_index],
                # "%H%M%S") == datetime.datetime.strptime(
                #     record.time, "%H%M%S"):
                # 文字列の長さによって付け足す
                # if len(psg_data["time"].iloc[_index]) == 5:
                #     __psg_time = "0" + psg_data["time"].iloc[_index]
                # if __psg_time == record.time:
                #     record.ss = psg_data["ss"].iloc[_index]
                #     return True

                # 公式に合致しない場合は，愚直に全体を探索
                # else:
                for counter, psg_time in enumerate(psg_data["time"]):
                    try:
                        assert type(psg_time) == str
                    except Exception:
                        print("catcher")
                    if len(psg_time) == 7:
                        __psg_time = "0" + psg_time
                    else:
                        __psg_time = psg_time
                    # if datetime.datetime.strptime(psg_time,
                    # "%H%M%S") == datetime.datetime.strptime(
                    #     record.time, "%H%M%S"):
                    if __psg_time == record.time:
                        record.ss = psg_data["ss"].iloc[counter]
                        return True

                    # タニタの時刻がすべてのpsgのデータに合致しないとき
                    # タニタの時間がpsgと比較して早すぎると起こる可能性あり（比較的最初に起こる）
                    # 逆にタニタのデータが多すぎると起こる可能性あり
                    # if record.ss == None:
                    #     print(PyColor.RED,
                    #           "record.time did not match any rule",
                    #           PyColor.END)

            # NOTE : noneがキャッチされる場所
            # （たくさんNoneがキャッチされている場合は無駄にオブジェクトを作成している可能性がある）
            # if record.ss == None:
            #     print(PyColor.RED_FLASH,
            #           "note here",
            #           PyColor.END)

        for start_point, record in tqdm(zip(start_points_list, records)):
            _make(start_point=start_point, record=record)
            has_match = _match(record=record, start_point=start_point)
            if not has_match:
                break
        from collections import Counter

        data = [record.ss for record in records]
        print(Counter(data))

        return records

    def makeSpectrogram(
        self, tanita_data, psg_data, sampleLen=1024, timeLen=128
    ):
        loopLen = int((len(tanita_data) - sampleLen) / 4)  # fft ができる回数
        recordLen = int(
            loopLen / timeLen
        )  # record オブジェクトを作る回数（128回のfftに関して1回作る）
        records = multipleRecords(recordLen)

        def _make():
            for i, record in enumerate(tqdm(records)):
                spectroGram = list()
                stop = 0
                for k in range(timeLen):
                    start = k * 4 + (4 * timeLen) * i
                    stop = start + sampleLen
                    if stop > len(tanita_data):
                        print(
                            f"stop index is {stop}.",
                            "this is out of range",
                            f"tanita data {len(tanita_data)}",
                        )
                        break
                    amped = (
                        hamming(len(tanita_data["sensor1"][start:stop]))
                        * tanita_data["sensor1"][start:stop]
                    )
                    fft = np.fft.fft(amped) / (
                        len(tanita_data["sensor1"][start:stop]) / 2.0
                    )
                    fft = 20 * np.log10(np.abs(fft))
                    fft = list(
                        fft[
                            : int(
                                len(tanita_data["sensor1"][start:stop]) / 2.0
                            )
                        ]
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

    def makeCepstrogram(
        self, tanita_data, psg_data, sampleLen=1024, timeLen=128
    ):
        loopLen = int((len(tanita_data) - sampleLen) / 4)  # fft ができる回数
        recordLen = int(
            loopLen / timeLen
        )  # record オブジェクトを作る回数（128回のfftに関して1回作る）
        records = multipleRecords(recordLen)

        def _make():
            for i, record in enumerate(tqdm(records)):
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
    from pre_process.tanita_reader import TanitaReader
    from pre_process.psg_reader import PsgReader

    CD = CreateData()
    tanita = TanitaReader("H_Hayashi")
    tanita.readCsv()
    psg = PsgReader("H_Hayashi")
    psg.readCsv()
    record = Record()
    data = tanita.df["val"][0 : 1024 + 511]
    spectroGram = CD.makeSpectrogram(tanita_data=tanita.df, psg_data=psg.df)
    plt.imshow(record.spectrogram)
