from dataclasses import dataclass

import numpy as np


@dataclass
class Record(object):
    time: str = ""
    spectrum_raw: np.ndarray = np.array([])
    spectrum: np.ndarray = np.array([])
    cepstrum: np.ndarray = np.array([])
    spectrogram_raw: np.ndarray = np.array([])
    spectrogram: np.ndarray = np.array([])
    wavelet_raw: np.ndarray = np.array([])
    wavelet: np.ndarray = np.array([])
    ss: int = None


# 複数レコードの作成
make_mul_records = lambda num: [Record() for _ in range(num)]

# TODO: 削除予定（上のラムダ式に移行して問題がなければ削除）
# def multipleRecords(num):
#     return [Record() for _ in range(num)]


if __name__ == "__main__":
    # Recordオブジェクト内の要素を出力する
    from data_analysis.py_color import PyColor

    record = Record()
    records = make_mul_records(10)
    for key, val in record.__dict__.items():
        print(PyColor.GREEN, f"key : {key}, ", f"val : {val}", PyColor.END)
