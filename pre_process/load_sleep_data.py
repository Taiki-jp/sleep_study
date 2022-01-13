import os
import pickle
import sys

# 以前のrecordの復旧のために必要
from json import load
from typing import List

from data_analysis.py_color import PyColor
from pre_process.my_env import MyEnv
from pre_process.record import Record


# 前処理後の睡眠データを読み込むためのメソッドを集めたクラス
class LoadSleepData:
    def __init__(
        self,
        data_type: str,
        cleansing_type: str,
        verbose: int = 0,
        fit_pos: str = "",
        kernel_size: int = 0,
        is_previous: bool = False,
        stride: int = 0,
        is_normal: bool = False,
        hostkey: str = "",
        model_type: str = "",
    ):
        self.my_env = MyEnv(
            is_normal=is_normal,
            is_previous=is_previous,
            data_type=data_type,
            fit_pos=fit_pos,
            stride=stride,
            kernel_size=kernel_size,
            model_type=model_type,
            cleansing_type=cleansing_type,
        )
        self.sl = self.my_env.sl
        self.data_type = data_type
        self.verbose = verbose
        self.fit_pos = fit_pos
        self.kernel_size = kernel_size
        self.is_previous = is_previous
        self.stride = stride
        self.is_normal = is_normal

    def load_data(
        self,
        name: str = None,
        load_all: bool = False,
        pse_data: bool = False,
    ) -> List[Record]:
        # NOTE : pse_data is needed for avoiding to load data
        if pse_data:
            print("仮データのため、何も読み込みません")
            return None
        if load_all:
            print("*** すべての被験者を読み込みます（load_dataの引数:nameは無視します） ***")
            records = list()
            if self.is_previous:
                if self.is_normal:
                    subjects = self.sl.prev_names
                else:
                    subjects = self.sl.prev_sass
            else:
                if self.is_normal:
                    subjects = self.sl.foll_names
                else:
                    subjects = self.sl.foll_sass

            for name in subjects:
                path = self.my_env.set_processed_filepath(
                    subject=name,
                    fit_pos=self.fit_pos,
                    data_type=self.data_type,
                )
                print(PyColor.GREEN, f"{name} を読み込みます", PyColor.END)
                records.append(pickle.load(open(path, "rb")))
            return records
        else:
            print("実装し直してください")
            raise Exception("no impl")


if __name__ == "__main__":
    import os
    from collections import Counter

    import pandas as pd
    import rich

    load_sleep_data = LoadSleepData(
        data_type="spectrogram",
        verbose=0,
        fit_pos="middle",
        kernel_size=128,
        is_previous=False,
        stride=16,
        is_normal=True,
        cleansing_type="nothing",
    )
    data = load_sleep_data.load_data(
        load_all=True,
    )
    # 各被験者について睡眠段階の量をチェック
    _df = None
    targets = load_sleep_data.sl.foll_names
    try:
        assert len(data) == len(targets)
        rich.print(
            f"=== data length {len(data)} equal with targets length {len(targets)} ==="
        )
    except AssertionError:
        print(
            f"=== data length {len(data)} doesn't equal with targets length {len(targets)} ==="
        )
        sys.exit(1)

    for each_data, target in zip(data, targets):
        _, target = os.path.split(target)
        ss = [_record.ss for _record in each_data]
        rich.print(Counter(ss))
        d = dict(Counter(ss))
        df = pd.DataFrame.from_dict(d, orient="index", columns=[target])
        if _df is not None:
            _df = pd.concat([df, _df], axis=1)
        else:
            _df = df

    # csvに書き込み
    path = os.path.join(load_sleep_data.my_env.tmp_dir, "ss.csv")
    _df.to_csv(path)
