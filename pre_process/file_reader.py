import os
import pickle
import sys

from data_analysis.py_color import PyColor
from pre_process.my_env import MyEnv
from pre_process.subjects_list import SubjectsList


class FileReader(object):
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
    ) -> None:

        self.sl: SubjectsList = SubjectsList()
        self.my_env: MyEnv = MyEnv(
            is_normal,
            is_previous,
            data_type,
            fit_pos,
            stride,
            kernel_size,
            model_type,
            cleansing_type,
        )
        self.ppi = self.my_env.ppi
        self.sl.load()

    # ファイルを読み込むためにファイルパスと被験者名を指定する
    def load(self, name: str = "", path_list: list = [], verbose: int = 0):

        # nameが空のときはエラーハンドル
        if name == "":
            print(PyColor.RED_FLASH, "nameを指定してください", PyColor.END)
            sys.exit(1)

        filepath = self.my_env.pre_processed_dir
        filepath = os.path.join(filepath, *path_list)

        def _load():

            if verbose == 0:
                print(PyColor.YELLOW, f"{filepath}を読み込んでいます...", PyColor.END)
            elif verbose == 1:
                print(PyColor.YELLOW, f"被験者{name}を読み込んでいます...", PyColor.END)
            else:
                pass
            return pickle.load(open(filepath, "rb"))

        return _load()


if __name__ == "__main__":
    pass
