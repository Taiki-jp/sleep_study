import datetime
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
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
    def get_subjects(self) -> list:
        return list(self.json_dict.keys())

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

    def can_cast_int(self, string: str):
        try:
            int(string)
            return True
        except:
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
    from data_analysis.py_color import PyColor

    sl = SubjectsInfo()
    sl.show_summary()
