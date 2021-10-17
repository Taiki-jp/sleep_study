import os
import json
import sys


class JsonBase(object):
    def __init__(self, json_filename: str) -> None:
        self.json_file: str = os.path.join(
            os.environ["git"],
            "sleep_study",
            "pre_process",
            json_filename,
        )
        self.json_dict: dict = {}
        self.prev_names: list = []
        self.prev_sass: list = []
        self.foll_names: list = []
        self.foll_sass: list = []

    def load(self) -> None:
        with open(self.json_file) as f:
            self.json_dict = json.load(f)

    def dump(self, keys: list, value: str) -> None:
        # 辞書であり，keysが入っている間はキーを入れる
        key_len = len(keys)
        if key_len > 6:
            print(PyColor().RED_FLASH, "キーは4までの長さまでしか実装されていません", PyColor().END)
            sys.exit(1)
        # TODO : もっと賢い書き方無いかな？
        if key_len == 1:
            self.json_dict[keys[0]] = value
        elif key_len == 2:
            self.json_dict[keys[0]][keys[1]] = value
        elif key_len == 3:
            self.json_dict[keys[0]][keys[1]][keys[2]] = value
        elif key_len == 4:
            self.json_dict[keys[0]][keys[1]][keys[2]][keys[3]] = value
        elif key_len == 5:
            self.json_dict[keys[0]][keys[1]][keys[2]][keys[3]][keys[4]] = value
        elif key_len == 6:
            self.json_dict[keys[0]][keys[1]][keys[2]][keys[3]][keys[4]][
                keys[5]
            ] = value
        with open(self.json_file, "w") as f:
            json.dump(self.json_dict, f, indent=4)

    # 複数リスト => 要素を辞書にまとめたリストを作成するメソッド
    def make_list_of_dict_from_mul_list(self, *args) -> list:
        first_list = self.json_dict[args[0]][args[1]][args[2]][args[3]][
            args[4]
        ]["no_cleansing"]
        second_list = self.json_dict[args[0]][args[1]][args[2]][args[3]][
            args[4]
        ]["positive_cleansing"]
        third_list = self.json_dict[args[0]][args[1]][args[2]][args[3]][
            args[4]
        ]["negative_cleansing"]

        # マップ関数によって iterable に変換してる kigasuru
        mapped = map(
            lambda x, y, z: dict(nothing=x, positive=y, negative=z),
            first_list,
            second_list,
            third_list,
        )
        return list(mapped)


if __name__ == "__main__":
    from data_analysis.py_color import PyColor

    jb = JsonBase("pre_processed_id.json")
    jb.load()
    for key, val in jb.__dict__.items():
        print(PyColor.GREEN, "key, ", key, "val, ", val, PyColor.END)
    jb.dump(keys=["spectrum", "middle", "stride_1"], value="")
