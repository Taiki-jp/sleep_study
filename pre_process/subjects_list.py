import sys, os, glob, json
from pre_process.json_base import JsonBase

class SubjectsList(JsonBase):
    def __init__(self) -> None:
        super().__init__(json_filename="./subjects_list.json")
    
    def load(self) -> None:
        with open(self.json_file) as f:
            self.json_dict = json.load(f)
        self.prev_names = self.json_dict["prev_subjects_name"]
        self.prev_sass = self.json_dict["prev_sas_name"]
        self.foll_names = self.json_dict["following_subjects_name"]
        self.foll_sass = self.json_dict["following_sas_name"]
    
if __name__ == '__main__':
    from data_analysis.py_color import PyColor
    sl = SubjectsList()
    sl.load()
    for key, val in sl.__dict__.items():
        print(PyColor.GREEN,
            "key, ", key, 
            "val, ", val, 
            PyColor.END)

    