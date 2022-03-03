import json

from data_analysis.py_color import PyColor
from pre_process.json_base import JsonBase


class PreProcessedId(JsonBase):
    def __init__(self) -> None:
        super().__init__(json_filename="pre_processed_id.json")
        # jsonをいじるときに指定するキーはここから使う
        self.hostkey = self.get_hostkey()
        self.subject_type = ""
        self.data_type = ""
        self.fit_pos = ""
        self.stride = ""
        self.kernel = ""
        # self.__load()

    def get_dateid(self):
        return self.json_dict[self.hostkey][self.subject_type][self.data_type][
            self.fit_pos
        ][self.stride][self.kernel]

    def set_key(
        self,
        is_normal: bool,
        is_previous: bool,
        data_type: str,
        fit_pos: str,
        stride: int,
        kernel: int,
    ) -> None:
        self.subject_type = self.first_key_of_pre_process(
            is_normal=is_normal, is_prev=is_previous
        )
        self.data_type = data_type
        self.fit_pos = fit_pos
        self.stride = "stride_" + str(stride)
        self.kernel = "kernel_" + str(kernel)

    def dump(self, is_pre_dump: bool = False, value="demo") -> None:
        def __dump():
            self.json_dict[self.hostkey][self.subject_type][self.data_type][
                self.fit_pos
            ][self.stride][self.kernel] = value
            with open(self.json_file, "w") as f:
                json.dump(self.json_dict, f, indent=2)

        if is_pre_dump:
            print(
                PyColor().RED,
                f"comfirming whether dump into {self.json_file} is possipoble or not",
                PyColor().END,
            )
        else:
            print(
                PyColor().GREEN_FLASH,
                f"dumping into {self.json_file}",
                PyColor().END,
            )

        __dump()


if __name__ == "__main__":
    from data_analysis.py_color import PyColor

    ppi = PreProcessedId()
    ppi.load()
    for key, val in ppi.__dict__.items():
        print(PyColor.GREEN, "key, ", key, "val, ", val, PyColor.END)
    print(PyColor.YELLOW, ppi.prev_datasets, PyColor.END)
