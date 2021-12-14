import json
import sys

from pre_process.json_base import JsonBase


class ModelId(JsonBase):
    def __init__(self) -> None:
        super().__init__(json_filename="model_id.json")
        self.hostkey = self.get_hostkey()
        self.subject_type = ""
        self.data_type = ""
        self.fit_pos = ""
        self.stride = ""
        self.fit_pos = ""
        self.load()

    def get_ppi(self):
        return self.json_dict[self.hostkey][self.subject_type][self.data_type][
            self.fit_pos
        ][self.stride][self.fit_pos]

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
        self.fit_pos = "kernel_" + str(kernel)

    def dump(self, is_pre_dump: bool) -> None:
        def __dump():
            self.json_dict[self.hostkey][self.data_type][self.model_type][
                self.fit_pos
            ][self.stride][self.kernel]
            with open(self.json_file, "w") as f:
                json.dump(self.json_dict, f, indent=2)

        if is_pre_dump:
            print(
                PyColor().RED_FLASH,
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

    model_id = ModelId()
    model_id.load()
    for depth_counter, (key, val) in enumerate(model_id.json_dict.items()):
        # print(PyColor.GREEN, "key, ", key, "val, ", val, PyColor.END)
        print(depth_counter)
