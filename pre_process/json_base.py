import json
import os
import sys

from data_analysis.py_color import PyColor


class JsonBase(object):
    def __init__(self, json_filename: str) -> None:
        self.json_file: str = os.path.join(
            os.environ["git"],
            "sleep_study",
            "env",
            json_filename,
        )
        self.json_dict: dict = {}
        self.prev_names: list = []
        self.prev_sass: list = []
        self.foll_names: list = []
        self.foll_sass: list = []

    def load(self) -> None:
        with open(self.json_file) as f:
            self.json_dict = json.load(f)

    def dump(self, keys: list, value: str, is_pre_dump: bool = False) -> None:
        if is_pre_dump:
            print(PyColor().RED_FLASH, "jsonに出力できるか確認します", PyColor().END)
        # 辞書であり，keysが入っている間はキーを入れる
        key_len = len(keys)
        if key_len > 7:
            print(PyColor().RED_FLASH, "キーは7までの長さまでしか実装されていません", PyColor().END)
            sys.exit(1)
        # TODO : もっと賢い書き方無いかな？
        if key_len == 1:
            self.json_dict[keys[0]] = value
        elif key_len == 2:
            self.json_dict[keys[0]][keys[1]] = value
        elif key_len == 3:
            self.json_dict[keys[0]][keys[1]][keys[2]] = value
        elif key_len == 4:
            self.json_dict[keys[0]][keys[1]][keys[2]][keys[3]] = value
        elif key_len == 5:
            self.json_dict[keys[0]][keys[1]][keys[2]][keys[3]][keys[4]] = value
        elif key_len == 6:
            self.json_dict[keys[0]][keys[1]][keys[2]][keys[3]][keys[4]][
                keys[5]
            ] = value
        elif key_len == 7:
            self.json_dict[keys[0]][keys[1]][keys[2]][keys[3]][keys[4]][
                keys[5]
            ][keys[6]] = value
        with open(self.json_file, "w") as f:
            json.dump(self.json_dict, f, indent=2)

    # 複数リスト => 要素を辞書にまとめたリストを作成するメソッド
    def make_list_of_dict_from_mul_list(self, *args) -> list:
        first_list = self.json_dict[args[0]][args[1]][args[2]][args[3]][
            args[4]
        ][args[5]]["no_cleansing"]
        second_list = self.json_dict[args[0]][args[1]][args[2]][args[3]][
            args[4]
        ][args[5]]["positive_cleansing"]
        # third_list = self.json_dict[args[0]][args[1]][args[2]][args[3]][
        #     args[4]
        # ]["negative_cleansing"]

        # マップ関数によって iterable に変換してる kigasuru
        # TODO: 最大のものに合わせてループを繰り返すようにする
        mapped = map(
            lambda x, y, z: dict(nothing=x, positive=y, negative=z),
            first_list,
            second_list,
            first_list,
        )

        return list(mapped)

    # per_processed_id.jsonのフォーマット作成
    def make_pre_processed_id_format(self) -> None:
        # キーを指定して辞書を作成するために，同じ階層の辞書をとりあえず作る
        os_version = ["win_249", "spica-2nd"]
        dataset = ["normal_prev", "normal_follow", "sas_prev", "sas_follow"]
        preprocess_type = ["spectrum", "spectrogram", "cepstrum"]
        ss_pos = ["bottom", "middle", "top"]
        stride = ["stride_" + str(i) for i in (1, 4, 16, 480, 1024)]
        kernel = ["kernel_" + str(i) for i in (512, 1024)]

        # 辞書の初期値のみをループとは別に作成しておく
        kernel_d = {_kernel: "" for _kernel in kernel}
        # リストの後ろにあるものほど、後にループ処理がかかるのでJsonの浅い場所のキーとなる
        list4loop = [stride, ss_pos, preprocess_type, dataset, os_version]
        for _list4loop in list4loop:
            merged_d = {__list4loop: kernel_d for __list4loop in _list4loop}
            kernel_d = merged_d.copy()
        # jsonの中身の書き換え
        self.json_dict = merged_d
        # jsonfileへの書き込み
        try:
            with open(self.json_file, "w") as f:
                json.dump(self.json_dict, f, indent=2)
        except Exception:
            print("jsonへの書き込みに失敗しました")
            sys.exit(1)

    # model_id.jsonのフォーマット作成
    def make_model_id_format(self) -> None:
        # キーを指定して辞書を作成するために，同じ階層の辞書をとりあえず作る
        os_version = ["win_249", "spica-2nd"]
        dataset = [
            "normal_prev",
            "normal_follow",
            "sas_prev",
            "sav_normal",
            "unused_set",
        ]
        model_type = ["dnn", "enn"]
        preprocess_type = ["spectrum", "spectrogram"]
        ss_pos = ["bottom", "middle", "top"]
        stride = ["stride_" + str(i) for i in (1, 4, 16, 480, 1024)]
        kernel = ["kernel_" + str(i) for i in (256, 512, 1024)]
        cleansing_type = [
            "no_cleansing",
            "positive_cleansing",
            "negative_cleansing",
        ]

        # 辞書の初期値のみをループとは別に作成しておく
        cleansing_type_d = {
            _cleansing_type: [] for _cleansing_type in cleansing_type
        }
        list4loop = [
            kernel,
            stride,
            ss_pos,
            preprocess_type,
            model_type,
            dataset,
            os_version,
        ]
        for _list4loop in list4loop:
            merged_d = {
                __list4loop: cleansing_type_d for __list4loop in _list4loop
            }
            cleansing_type_d = merged_d.copy()
        # jsonの中身の書き換え
        self.json_dict = merged_d
        # jsonfileへの書き込み
        try:
            with open(self.json_file, "w") as f:
                json.dump(self.json_dict, f, indent=2)
        except Exception:
            print("jsonへの書き込みに失敗しました")
            sys.exit(1)

    # main_param.jsonのフォーマット作成
    # NOTE: 使わない
    def make_main_param_format(self) -> None:
        # キーを指定して辞書を作成するために，同じ階層の辞書をとりあえず作る
        dataset = [
            "main",
            "main_custom",
            "pre_process_main",
            "data_analysis_main",
            "data_merging",
            "data_selecting",
        ]
        params = [
            "data_type",
            "fit_pos",
            "stride",
            "test_run",
            "epochs",
            "has_attention",
            "pse_data",
            "has_inception",
            "is_previous",
            "is_normal",
            "has_dropout",
            "is_enn",
            "is_mul_layer",
            "has_nrem2_bias",
            "has_rem_bias",
            "dropout_rate",
            "batch_size",
            "n_class",
            "kernel_size",
            "stride",
            "sample_size",
        ]

        # 辞書の初期値のみをループとは別に作成しておく
        kernel_d = {_params: [] for _params in params}
        list4loop = [dataset]
        for _list4loop in list4loop:
            merged_d = {__list4loop: kernel_d for __list4loop in _list4loop}
            kernel_d = merged_d.copy()
        # jsonの中身の書き換え
        self.json_dict = merged_d
        # jsonfileへの書き込み
        try:
            with open(self.json_file, "w") as f:
                json.dump(self.json_dict, f, indent=2)
        except Exception:
            print("jsonへの書き込みに失敗しました")
            sys.exit(1)

    # フォーマット作成のメタメソッド
    def make_format(self, json: str) -> None:
        if json == "pre_processed_id":
            self.make_pre_processed_id_format()
        elif json == "model_id":
            self.make_model_id_format()
        else:
            print("実装まだです．飛ばします")
            return

    # pre_processの第一キーを返す
    def first_key_of_pre_process(self, is_normal: bool, is_prev: bool) -> str:
        if is_normal:
            if is_prev:
                key_name = "normal_prev"
            else:
                key_name = "normal_follow"
        else:
            if is_prev:
                key_name = "sas_prev"
            else:
                key_name = "sas_follow"
        return key_name


if __name__ == "__main__":
    filenames = [
        "pre_processed_id.json",
        "model_id.json",
        # "my_color.json",
        # "ss.json",
        # "subjects_list.json",
    ]
    for _filenames in filenames:
        jb = JsonBase(_filenames)
        jb.load()

        # フォーマット作成のテスト
        jb.make_format(json=_filenames.split(".")[0])
