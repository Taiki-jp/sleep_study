from pre_process.json_base import JsonBase


class MainParamReader(JsonBase):
    def __init__(self, json_filename: str = "main_param.json") -> None:
        super().__init__(json_filename)
        self.main_setting = self.json_dict["main"]
        self.denn_ensemble = self.json_dict["denn_ensemble"]
        self.attention_ipynb = self.json_dict["attention_ipynb"]


if __name__ == "__main__":
    MPR = MainParamReader()
    print(MPR)
