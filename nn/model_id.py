import json
from pre_process.json_base import JsonBase
import sys


class ModelId(JsonBase):
    def __init__(self) -> None:
        super().__init__(json_filename="./model_id.json")

    def load(self) -> None:
        with open(self.json_file) as f:
            self.json_dict = json.load(f)

    def dump(self, keys: list, value: str) -> None:
        # 辞書であり，keysが入っている間はキーを入れる
        key_len = len(keys)
        if key_len > 4:
            print(PyColor.RED_FLASH, f"キーは4までの長さまでしか実装されていません", PyColor.END)
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
        with open(self.json_file, "w") as f:
            json.dump(self.json_dict, f, indent=4)


if __name__ == "__main__":
    from data_analysis.py_color import PyColor

    ppi = PreProcessedId()
    ppi.load()
    for key, val in ppi.__dict__.items():
        print(PyColor.GREEN, "key, ", key, "val, ", val, PyColor.END)
    print(PyColor.YELLOW, ppi.prev_datasets, PyColor.END)
