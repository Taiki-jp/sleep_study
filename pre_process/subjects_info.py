from __future__ import annotations

import datetime
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import pandas as pd
from _collections_abc import dict_keys
from rich import print

from pre_process.json_base import JsonBase


class SubjectsInfo(JsonBase):
    def __init__(self) -> None:
        super().__init__(json_filename="subjects_info.json")

    # 被験者情報のサマリを出力するメソッド
    def show_summary(self) -> None:
        # サマリ用の辞書の初期化
        summary_d = {str(i * 10) + "'s": 0 for i in range(1, 9)}
        age_d = self.get_age()
        # 10~70代の数をカウントする
        for key, value in age_d.items():
            for i in range(1, 9):
                if int(value) < (i + 1) * 10:
                    summary_d[str(i * 10) + "'s"] += 1
                    break
        print(summary_d)

    # 被験者のリストを取得するメソッド
    def get_subjects(self) -> List[str]:
        return list(self.json_dict.keys())

    def get_ss_time(self) -> Dict[str, float]:
        json_cp = self.json_dict.copy()
        name_ss_time_d = {}
        for __subject, __info in json_cp.items():
            name_ss_time_d.update({__subject: __info["sleeping_time"]})

        return name_ss_time_d

    def get_sex(self) -> Dict[str, str]:
        json_cp = self.json_dict.copy()
        name_sex_d = {}
        for __subject, __info in json_cp.items():
            name_sex_d.update({__subject: __info["sex"]})

        return name_sex_d

    # 被験者の実験当時の年齢を返すメソッド
    def get_age(self) -> Dict[str, str]:
        json_cp = self.json_dict.copy()
        # 1. 年齢が埋められている被験者はそのまま取り出す
        added_key = [
            __keyname
            for __keyname in json_cp.keys()
            if bool(self.json_dict[__keyname]["age"])
        ]
        added_d_by_age = {__key: json_cp[__key]["age"] for __key in added_key}
        # 1の条件に引っかかった被験者は除く
        for __keyname in self.get_subjects():
            if __keyname in added_key:
                del json_cp[__keyname]
        added_key = [
            __keyname
            for __keyname in json_cp.keys()
            if self.can_cast_int(__keyname[:6])
            and bool(json_cp[__keyname]["birth"][:6])
        ]
        added_d_by_birth = {__key: self.calc_age(__key) for __key in added_key}
        # 辞書の結合
        added_d_by_age.update(added_d_by_birth)
        return added_d_by_age

    def can_cast_int(self, string: str) -> bool:
        try:
            int(string)
            return True
        except Exception:
            return False

    def convert_str2datetime(self, time_str: str) -> datetime.datetime:
        return datetime.datetime.strptime(time_str, "%Y%m%d")

    def calc_age(self, keyname: str) -> str:
        birth = self.convert_str2datetime(self.json_dict[keyname]["birth"])
        keyname = self.convert_key2exact_date(keyname)
        experiment_date = self.convert_str2datetime(keyname)
        time_delta = experiment_date - birth
        return str(time_delta.days // 365)

    def convert_key2exact_date(self, keyname: str) -> str:
        # 2014 ~ 2019年のデータは20がついていないのでつける
        if keyname[:2] in ["14", "15", "16", "17", "18", "19"]:
            return "20" + keyname[:6]
        # 2020移行のデータは2020がついているのでそのまま返す
        elif keyname[:2] in ["20"]:
            return keyname[:6]
        else:
            raise Exception


if __name__ == "__main__":
    import os

    import pandas as pd

    from data_analysis.py_color import PyColor

    sl = SubjectsInfo()
    tmp = sl.get_ss_time()
    # NOTE: 辞書の中身がstrで入っているのでfrom_dictは使えない
    # df = pd.DataFrame.from_dict(tmp)
    df = pd.DataFrame(
        tmp,
        index=[
            "i",
        ],
    )
    filepath = os.path.join(os.environ["sleep"], "tmp", "time_list.csv")
    df.to_csv(filepath)

    # sl.show_summary()
