import datetime
import json

from pre_process.json_base import JsonBase


class SubjectsInfo(JsonBase):
    def __init__(self) -> None:
        super().__init__(json_filename="subjects_info.json")

    # 被験者情報のサマリを出力するメソッド
    def get_summary(self) -> None:
        print("data num is ", len(self.json_dict))

    # 被験者のリストを取得するメソッド
    def get_subjects(self) -> list:
        return list(self.json_dict.keys())

    # 被験者の実験当時の年齢を返すメソッド
    def get_age(self) -> dict:
        json_cp = self.json_dict.copy()
        # 1. 年齢が埋められている被験者はそのまま取り出す
        added_key = [
            json_cp[__keyname]
            for __keyname in json_cp.keys()
            if bool(self.json_dict[__keyname]["age"])
        ]
        added_d = {__key: json_cp[__key]["age"] for __key in added_key}
        # 1の条件に引っかかった被験者は除く
        for __keyname in self.get_subjects():
            if __keyname in added_key:
                del json_cp[__keyname]
        # 2. 最初の6文字が生年月日でキーがスタートしている かつ birthが存在するモノのみでカウント
        # 2010/1/1 ~ 2021/12/1まではキャッチする
        time_limit = ("2010-01-01", "2021-12-01")

        def __conv_str2datetime(time_str: str):
            return datetime.datetime.strptime(time_str, "%Y-%m-%d")

        time_limit = tuple(map(__conv_str2datetime, time_limit))
        # TODO:
        # - datetimeオブジェクトに変換可能なもののみ最初の6文字の返還を行う
        # - 指定の範囲に入っていればリストにキーを追加する
        # 追加したキーから辞書を作成し、手順(1)で作成した辞書とマージする

        # for __keynaem in json_cp.keys():
        #     time_delta =
        # added_key = [json_cp[__keyname] for __keyname in json_cp.keys() if self.json_dict[__keyname][:6]]
        # added_key = []
        # for __subject in json_cp:


if __name__ == "__main__":
    from data_analysis.py_color import PyColor

    sl = SubjectsInfo()
    sl.load()
    print(sl.get_subjects())
    print(sl.get_summary())
