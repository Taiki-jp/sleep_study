import json
import sys

from pre_process.json_base import JsonBase


class ModelId(JsonBase):
    def __init__(self) -> None:
        super().__init__(json_filename="model_id.json")


if __name__ == "__main__":
    from data_analysis.py_color import PyColor

    model_id = ModelId()
    model_id.load()
    for depth_counter, (key, val) in enumerate(model_id.json_dict.items()):
        # print(PyColor.GREEN, "key, ", key, "val, ", val, PyColor.END)
        print(depth_counter)
