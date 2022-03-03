from __future__ import annotations

import json
from ctypes import ArgumentError
from typing import List

from pre_process.json_base import JsonBase


class SubjectsList(JsonBase):
    def __init__(self) -> None:
        super().__init__(json_filename="./subjects_list.json")

    def load(self) -> None:
        with open(self.json_file) as f:
            self.json_dict = json.load(f)
        self.prev_names = self.json_dict["prev_subjects_name"]
        self.prev_sass = self.json_dict["prev_sas_name"]
        self.foll_names = self.json_dict["following_subjects_name"]
        self.foll_sass = self.json_dict["following_sas_name"]

    # 名前のリストを返す
    def set_name_list(self, is_previous: bool, is_normal: bool) -> List[str]:
        if is_previous:
            if is_normal:
                names = self.prev_names
            else:
                names = self.prev_sass
        else:
            if is_normal:
                names = self.foll_names
            else:
                names = self.foll_sass
        return names

    # 無視する名前のリストを返す
    def get_ignoring_list(self, option) -> List[str]:
        if option == "nr2":
            names = self.json_dict["ignoring_list"]["nr2_rate"]
        else:
            raise ArgumentError
        return names


if __name__ == "__main__":
    from data_analysis.py_color import PyColor

    sl = SubjectsList()
    sl.load()
    for key, val in sl.__dict__.items():
        print(PyColor.GREEN, "key, ", key, "val, ", val, PyColor.END)
