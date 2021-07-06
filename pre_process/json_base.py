import sys, os, glob, json

class JsonBase(object):
    def __init__(self, json_filename : str) -> None:
        self.json_file : str = os.path.join(os.environ["userprofile"],
                                            "git",
                                            "sleep_study",
                                            "pre_process",
                                            json_filename)
        self.json_dict : dict = {}
        self.prev_names : list = []
        self.prev_sass : list = []
        self.foll_names : list = []
        self.foll_sass : list = []
    
    def load(self) -> None:
        with open(self.json_file) as f:
            self.json_dict = json.load(f)
        # 各jsonファイルで下のキーワード，変数名を変更する
        # self.prev_names = self.json_dict["prev_subjects_name"]
        # self.prev_sass = self.json_dict["prev_sas_name"]
        # self.foll_names = self.json_dict["following_subjects_name"]
        # self.foll_sass = self.json_dict["following_sas_name"]
    
if __name__ == '__main__':
    from data_analysis.py_color import PyColor
    sl = JsonBase("./pre_process.json")
    sl.load()
    for key, val in sl.__dict__.items():
        print(PyColor.GREEN,
              "key, ", key, 
              "val, ", val, 
              PyColor.END)

    