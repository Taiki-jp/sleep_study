from pre_process.subjects_list import SubjectsList
from pre_process.file_reader import FileReader
from pre_process.my_env import MyEnv

# 前処理後の睡眠データを読み込むためのメソッドを集めたクラス
class LoadSleepData:
    def __init__(self, data_type, verbose=0, n_class=5):
        self.fr = FileReader()
        self.sl = SubjectsList()
        self.env = MyEnv()
        self.data_type = data_type
        self.verbose = verbose

    def load_data(
        self,
        name: str = None,
        load_all: bool = False,
        pse_data: bool = False,
        fit_pos: str = None,
        kernel_size: int = 0,
        is_previous: bool = False,
        data_type: str = "",
        stride: int = 0,
        is_normal: bool = False,
    ):
        # NOTE : pse_data is needed for avoiding to load data
        if pse_data:
            print("仮データのため、何も読み込みません")
            return None
        if load_all:
            print("*** すべての被験者を読み込みます（load_dataの引数:nameは無視します） ***")
            records = list()
            if is_previous:
                if is_normal:
                    subjects = self.sl.prev_names
                else:
                    subjects = self.sl.prev_sass
            else:
                if is_normal:
                    subjects = self.sl.foll_names
                else:
                    subjects = self.sl.foll_sass

            for name in subjects:
                path = self.env.set_processed_filepath(
                    is_previous=is_previous,
                    data_type=data_type,
                    subject=name,
                    stride=stride,
                    fit_pos=fit_pos,
                    kernel_size=kernel_size,
                )
                records.extend(self.fr.load)
            return records
        else:
            print("*** 一人の被験者を読み込みます ***")
            return self.fr.load_normal(
                name=name, verbose=self.verbose, data_type=self.data_type
            )


if __name__ == "__main__":
    from collections import Counter

    load_sleep_data = LoadSleepData(data_type="spectrum", verbose=0, n_class=5)
    data = load_sleep_data.load_data(
        load_all=True, pse_data=False, name=None, fit_pos="bottom"
    )
    # 各被験者について睡眠段階の量をチェック
    for each_data in data:
        ss = [record.ss for record in each_data]
        print(Counter(ss))
