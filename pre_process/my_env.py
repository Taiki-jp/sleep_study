import os, sys
from pre_process.file_reader import FileReader
import datetime

class MyEnv():
    def __init__(self) -> None:
        self.project_dir : str = os.environ["sleep"]
        self.figure_dir : str = os.path.join(self.project_dir, "figures")
        self.video_dir : str = os.path.join(self.project_dir, "videos")
        self.tmp_dir : str = os.path.join(self.project_dir, "tmps")
        self.analysis_dir : str = os.path.join(self.project_dir, "analysis")
        self.models_dir : str = os.path.join(self.project_dir, "models")
        self.pre_processed_dir : str = os.path.join(self.project_dir, "datas", "pre_processed_data")
        self.id : str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.name_list = self.fr.sl.name_list
        self.name_dict = self.fr.sl.name_dict
        self.ss_list = self.fr.sl.ss_list
      