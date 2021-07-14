from json import load
import pickle
from pre_process.subjects_list import SubjectsList
from pre_process.file_reader import FileReader
from pre_process.my_env import MyEnv
from data_analysis.py_color import PyColor

# 前処理後の睡眠データを読み込むためのメソッドを集めたクラス
class LoadSleepData:
    def __init__(
        self,
        data_type: str,
        verbose: int = 0,
        fit_pos: str = "",
        kernel_size: int = 0,
        is_previous: bool = False,
        stride: int = 0,
        is_normal: bool = False,
    ):
        self.fr = FileReader()
        self.sl = self.fr.sl
        self.my_env = self.fr.my_env
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
    ):
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
                    is_previous=self.is_previous,
                    subject=name,
                    stride=self.stride,
                    fit_pos=self.fit_pos,
                    kernel_size=self.kernel_size,
                    data_type=self.data_type,
                )
                print(PyColor.GREEN, f"{name} を読み込みます", PyColor.END)
                records.append(pickle.load(open(path, "rb")))
            return records
        else:
            print("*** 一人の被験者を読み込みます ***")
            return self.fr.load_normal(
                name=name, verbose=self.verbose, data_type=self.data_type
            )


if __name__ == "__main__":
    from collections import Counter
    import pandas as pd
    import os

    load_sleep_data = LoadSleepData(data_type="spectrum", verbose=0)
    data = load_sleep_data.load_data(
        load_all=True,
        pse_data=False,
        name=None,
        fit_pos="middle",
        kernel_size=1024,
        is_previous=False,
        stride=1024,
        is_normal=True,
    )
    # 各被験者について睡眠段階の量をチェック
    _df = None
    targets = load_sleep_data.sl.foll_names
    for each_data, target in zip(data, targets):
        _, target = os.path.split(target)
        ss = [record.ss for record in each_data]
        d = dict(Counter(ss))
        df = pd.DataFrame.from_dict(d, orient="index", columns=[target])
        if _df is not None:
            _df = pd.concat([df, _df], axis=1)
        else:
            _df = df

    # csvに書き込み
    path = os.path.join(os.environ["sleep"], "datas", "ss.csv")
    _df.to_csv(path)
