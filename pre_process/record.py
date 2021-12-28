from __future__ import annotations

from typing import List, NewType

import numpy as np


class Record(object):
    def __init__(self) -> None:
        self.time: str = ""
        self.spectrum_raw: np.ndarray = np.array([])
        self.spectrum: np.ndarray = np.array([])
        self.cepstrum: np.ndarray = np.array([])
        self.spectrogram_raw: np.ndarray = np.array([])
        self.spectrogram: np.ndarray = np.array([])
        self.wavelet_raw: np.ndarray = np.array([])
        self.wavelet: np.ndarray = np.array([])
        self.ss: int = None
        self.name: str = ""
        self.age: int = 0

    # 継承のことを考えるとRecordよりは__class__の方が良いが，mypyではサポートされていない
    # https://github.com/python/mypy/issues/4177
    @staticmethod
    def drop_none(records: List[Record]) -> List[Record]:
        records_cp = list()
        for _record in records:
            # リストから値を消すときは削除条件にマッチしないものを別配列に追加する方針を取る
            if _record.ss is not None:
                records_cp.append(_record)
        return records_cp


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
