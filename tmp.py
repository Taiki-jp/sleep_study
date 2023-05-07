from glob import glob
import json
import os

normal_f = os.path.join(
    os.environ["sleep"], "datas", "raw_data", "sas_following", "*"
)
normal_f = glob.glob(normal_f)
_abs_path_list = list()
for abs_path in normal_f:
    _, _file_name = os.path.split(abs_path)
    _abs_path_list.append(_file_name)
tmp_d = {"following_subjects_name": _abs_path_list}
with open("./_tmp.json", "w") as f:
    json.dump(tmp_d, f, indent=4)
