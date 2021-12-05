import json

from pre_process.json_base import JsonBase


class SubjectsInfo(JsonBase):
    def __init__(self) -> None:
        super().__init__(json_filename="subjects_list.json")


if __name__ == "__main__":
    from data_analysis.py_color import PyColor

    sl = SubjectsInfo()
    sl.load()
    for key, val in sl.__dict__.items():
        print(PyColor.GREEN, "key, ", key, "val, ", val, PyColor.END)
