import os, sys
from dataclasses import dataclass
import datetime

@dataclass
class MyEnv():
    project_dir : str = os.environ["sleep"]
    figure_dir : str = os.path.join(project_dir, "figures")
    video_dir : str = os.path.join(project_dir, "videos")
    tmp_dir : str = os.path.join(project_dir, "tmps")
    analysis_dir : str = os.path.join(project_dir, "analysis")
    models_dir : str = os.path.join(project_dir, "models")
    pre_processed_dir : str = os.path.join(project_dir, "datas", "pre_processed_data")
    id : str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
if __name__ == "__main__":
    from data_analysis.py_color import PyColor
    my_env = MyEnv()
    for key, val in my_env.__dict__.items():
        print(PyColor.GREEN,
              "key, ", key,
              "val, ", val,
              PyColor.END)
