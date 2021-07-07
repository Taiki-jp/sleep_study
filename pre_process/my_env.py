import os
import glob


class MyEnv:
    def __init__(self) -> None:
        self.project_dir: str = os.environ["sleep"]
        self.figure_dir: str = os.path.join(self.project_dir, "figures")
        self.video_dir: str = os.path.join(self.project_dir, "videos")
        self.tmp_dir: str = os.path.join(self.project_dir, "tmps")
        self.analysis_dir: str = os.path.join(self.project_dir, "analysis")
        self.models_dir: str = os.path.join(self.project_dir, "models")
        self.data_dir: str = os.path.join(self.project_dir, "datas")
        self.pre_processed_dir: str = os.path.join(
            self.project_dir, "datas", "pre_processed_data"
        )

    def set_raw_folder_path(self, is_normal: bool, is_previous: bool) -> list:
        if is_normal:
            if is_previous:
                root_abs = os.path.join(self.data_dir, "normal_previous", "*")
            else:
                root_abs = os.path.join(self.data_dir, "normal_following", "*")
        else:
            if is_previous:
                root_abs = os.path.join(self.data_dir, "sas_previous", "*")
            else:
                root_abs = os.path.join(self.data_dir, "sas_following", "*")
        return glob.glob(root_abs)

    # def set_processed_filepath(self,
    #                            data_type: str,
    #                            fit_pos: str) -> dict:
    #     if data_type == "previous_dataset":
    #         date_id =


if __name__ == "__main__":
    from data_analysis.py_color import PyColor

    my_env = MyEnv()
    for key, val in my_env.__dict__.items():
        print(PyColor.GREEN, "key, ", key, "val, ", val, PyColor.END)
