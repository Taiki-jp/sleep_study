import os
from typing import List

import pandas as pd

from data_analysis.py_color import PyColor


class CsvReader(object):
    def __init__(
        self,
        person_dir: str,
        file_name: str,
        verbose: int = 0,
        is_previous: bool = True,
        columns: list = list(),
        names: list = list(),
    ):
        self.person_dir: str = person_dir
        self.file_name: str = file_name
        self.df: pd.DataFrame = None
        self.verbose: int = verbose
        self.is_previous: bool = is_previous
        self.columns: str = columns
        self.names: List[str] = names

    def read_csv(self):
        file_path = os.path.join(self.person_dir, self.file_name)

        def _read():
            if self.verbose == 0:
                print(
                    PyColor.GREEN,
                    f"*** read {file_path} ***",
                    PyColor.END,
                )
            elif self.verbose == 1:
                _, name = os.path.split(self.person_dir)
                print(PyColor.GREEN, f"*** read {name} ***", PyColor.END)
            else:
                pass

            if bool(self.names):
                self.df = pd.read_csv(
                    file_path, usecols=self.columns, names=self.names, header=0
                )
            else:
                self.df = pd.read_csv(file_path, usecols=self.columns)

        return _read()


if __name__ == "__main__":

    csv_reader = CsvReader(
        "sleep", "datas/my_raw_data", "H_Hayashi", "signal_after.csv"
    )
    csv_reader.readCsv()
