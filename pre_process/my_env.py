import sys
import os
import glob
from pre_process.pre_processed_id import PreProcessedId


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
        self.raw_dir: str = os.path.join(self.data_dir, "raw_data")
        self.ppi = PreProcessedId()
        self.ppi.load()

    # 生データの被験者までのフォルダパスを指定する
    def set_raw_folder_path(self, is_normal: bool, is_previous: bool) -> list:
        if is_normal:
            if is_previous:
                root_abs = os.path.join(self.raw_dir, "normal_previous", "*")
            else:
                root_abs = os.path.join(self.raw_dir, "normal_following", "*")
        else:
            if is_previous:
                root_abs = os.path.join(self.raw_dir, "sas_previous", "*")
            else:
                root_abs = os.path.join(self.raw_dir, "sas_following", "*")
        _root_abs, _ = os.path.split(root_abs)
        if not os.path.exists(_root_abs):
            print("指定のパスが存在しません")
            sys.exit(2)
        return glob.glob(root_abs)

    def set_processed_filepath(
        self,
        is_previous: bool,
        data_type: str,
        subject: str,
        stride: int = 0,
        fit_pos: str = "",
        kernel_size: int = 0,
    ) -> dict:
        if is_previous:
            date_id = self.ppi.prev_datasets[data_type]["id"]
            path = os.path.join(
                self.pre_processed_dir,
                "previous_dataset",
                f"{subject}_{date_id}.sav",
            )
        else:
            date_id = self.ppi.json_dict[data_type][fit_pos][
                f"stride_{stride}"
            ][f"kernel_{kernel_size}"]
            path = os.path.join(
                self.pre_processed_dir,
                data_type,
                fit_pos,
                f"{subject}_{date_id}.sav",
            )
        if not os.path.exists(path):
            print(PyColor.RED_FLASH, f"{path}が存在しません", PyColor.END)
            sys.exit(3)
        return path


if __name__ == "__main__":
    from data_analysis.py_color import PyColor

    my_env = MyEnv()
    for key, val in my_env.__dict__.items():
        print(PyColor.GREEN, "key, ", key, "val, ", val, PyColor.END)
    target_list = my_env.set_processed_filepath(
        is_previous=False,
        data_type="spectrum",
        subject="140703_Li",
        stride=16,
        fit_pos="middle",
    )
    print(PyColor.YELLOW, target_list, PyColor.END)
