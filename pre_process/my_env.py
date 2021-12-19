import glob
import os
import sys

from data_analysis.py_color import PyColor
from nn.model_id import ModelId
from pre_process.pre_processed_id import PreProcessedId
from pre_process.subjects_info import SubjectsInfo


class MyEnv:
    def __init__(
        self,
        is_normal,
        is_previous,
        data_type,
        fit_pos,
        stride,
        kernel_size,
        model_type: str = "",
        cleansing_type: str = "",
    ) -> None:
        self.is_normal = is_normal
        self.is_previous = is_previous
        self.data_type = data_type
        self.fit_pos = fit_pos
        self.stride = stride
        self.kernel_size = kernel_size
        self.model_type = model_type
        self.cleansing_type = cleansing_type
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
        self.mi = ModelId()
        self.si = SubjectsInfo()
        self.set_jsonkey()

    # set json_keys as its member variables
    def set_jsonkey(self):
        self.ppi.set_key(
            is_normal=self.is_normal,
            is_previous=self.is_previous,
            data_type=self.data_type,
            fit_pos=self.fit_pos,
            stride=self.stride,
            kernel=self.kernel_size,
        )
        self.mi.set_key(
            is_normal=self.is_normal,
            is_previous=self.is_previous,
            data_type=self.data_type,
            fit_pos=self.fit_pos,
            stride=self.stride,
            kernel=self.kernel_size,
            model_type=self.model_type,
            cleansing_type=self.cleansing_type,
        )

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

    def get_ppi(self):
        return self.ppi.get_dateid()

    # 前処理後のファイルを指定して読み込む
    def set_processed_filepath(
        self,
        data_type: str,
        subject: str,
        fit_pos: str = "",
    ) -> str:
        date_id = self.get_ppi()
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
