import json

from pre_process.json_base import JsonBase


class PreProcessedId(JsonBase):
    def __init__(self) -> None:
        super().__init__(json_filename="pre_processed_id.json")

    def load(self) -> None:
        with open(self.json_file) as f:
            self.json_dict = json.load(f)
        self.hostkey = self.get_hostkey()


if __name__ == "__main__":
    from data_analysis.py_color import PyColor

    ppi = PreProcessedId()
    ppi.load()
    for key, val in ppi.__dict__.items():
        print(PyColor.GREEN, "key, ", key, "val, ", val, PyColor.END)
    print(PyColor.YELLOW, ppi.prev_datasets, PyColor.END)
