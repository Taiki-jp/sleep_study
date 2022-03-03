import json
import os
import socket
import sys
from typing import Any, Dict, List

from data_analysis.py_color import PyColor


class JsonBase:
    def __init__(self, json_filename: str) -> None:
        self.hostname: str = socket.gethostname()
        self.json_file: str = os.path.join(
            os.environ["git"],
            "sleep_study",
            "env",
            json_filename,
        )
        self.json_dict: Dict[str, Any] = {}
        self.prev_names: List[str] = []
        self.prev_sass: List[str] = []
        self.foll_names: List[str] = []
        self.foll_sass: List[str] = []
        # 各json形式で共通のキーはクラスメンバとして定義
        self.os_version: List[str] = [
            "win_249",
            "spica-2nd",
            "home-pc",
            "spica-4th",
        ]
        self.dataset: List[str] = [
            # "normal_prev",
            "normal_follow",
            # "sas_prev",
            # "sas_follow",
        ]
        self.ss_list: List[str] = ["wake", "rem", "nr1", "nr2", "nr3"]
        self.preprocess_type: List[str] = ["spectrogram"]
        self.ss_pos: List[str] = ["middle"]
        self.stride: List[str] = ["stride_16"]
        self.kernel: List[str] = ["kernel_" + str(i) for i in (128, 256)]
        self.added_key: List[str] = ["birth", "sex", "sleeping_time"]
        self.subjects_list: List[str] = [
            "140703_Li",
            "140711_Yamamoto",
            "140819_Kumazawa",
            "140820_Yasuda",
            "140821_Murakami",
            "140823_Murakami",
            "140825_Kashiwazaki",
            "140826_Ootsuji",
            "140828_Otsuji",
            "140830_Murakami",
            "140922_Kumazawa_M",
            "140925_Kashiwazaki",
            "140929_Kumazawa",
            "140930_Hayashi",
            "141001_Hayashi",
            "141002_Hayashi",
            "141003_Kashiwazaki_F",
            "141006_Hiromoto",
            "141014_Kumazawa_F",
            "141015_Hiromoto",
            "141024_Kawasaki_F",
            "141027_Kawasaki",
            "141029_Umezawa",
            "141030_Kumazawa",
            "141104_Takadama",
            "141124_Murata",
            "141127_Tatsumi",
            "141128_Sato_Minato",
            "141204_Tatebe",
            "141205_Fujitsuka",
            "141217_Sugimoto",
            "141219_Usui",
            "151104_Nagae",
            "151105_Nagae",
            "151106_Nagae",
            "151111_Kawasaki",
            "151112_Kawasaki",
            "151113_Kawasaki",
            "151118_Tomoko_Nagae",
            "151119_Tomoko_Nagae",
            "151120_Tanaka",
            "151126_Tanaka",
            "151127_Tanaka",
            "151201_Tomoko_Nagae",
            "151204_Umenai",
            "151207_Matsumoto",
            "151209_Umenai",
            "151210_Hoshino",
            "151214_Matsumoto",
            "151215_Ishii_Haruyuki",
            "151217_Hoshino",
            "151217_Ishii_Haruyuki",
            "190910_takadama_aki",
            "20190818_takano",
            "20190819_kobayashi",
            "20190820_murata",
            "20200721_maesuke",
            "20200723_yamane",
            "20200804_shiraishi",
            "20200814_shiraishi",
            "20200818_shiraishi",
            "20200819_maesuke",
            "20200819_shiraishi",
            "20200822_yamane",
            "20210130_iei",
            "20210131_ihou",
            "20210201_isei",
            "20210203_ihou",
            "20210204_kenka",
        ]
        self.model_type: List[str] = ["dnn", "enn"]
        self.cleansing_type: List[str] = [
            "no_cleansing",
            "positive_cleansing",
            "negative_cleansing",
        ]
        # 初期化時にファイルの読み込みを呼び出す
        self.__load()

    def __load(self) -> None:
        with open(self.json_file) as f:
            self.json_dict = json.load(f)

    def dump(
        self, keys: List[str] = [], value: str = "", is_pre_dump: bool = False
    ) -> None:
        def __dump():
            # 辞書であり，keysが入っている間はキーを入れる
            key_len = len(keys)
            if key_len > 8:
                print(
                    PyColor().RED_FLASH,
                    "キーは7までの長さまでしか実装されていません",
                    PyColor().END,
                )
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
                self.json_dict[keys[0]][keys[1]][keys[2]][keys[3]][
                    keys[4]
                ] = value
            elif key_len == 6:
                self.json_dict[keys[0]][keys[1]][keys[2]][keys[3]][keys[4]][
                    keys[5]
                ] = value
            elif key_len == 7:
                self.json_dict[keys[0]][keys[1]][keys[2]][keys[3]][keys[4]][
                    keys[5]
                ][keys[6]] = value
            elif key_len == 8:
                self.json_dict[keys[0]][keys[1]][keys[2]][keys[3]][keys[4]][
                    keys[5]
                ][keys[6]][keys[7]] = value
            with open(self.json_file, "w") as f:
                json.dump(self.json_dict, f, indent=2)

        if is_pre_dump:
            print(
                PyColor().RED_FLASH,
                f"comfirming whether dump into {self.json_file} is possipoble or not",
                PyColor().END,
            )
        else:
            print(
                PyColor().GREEN_FLASH,
                f"dumping into {self.json_file}",
                PyColor().END,
            )

        __dump()

    # 複数リスト => 要素を辞書にまとめたリストを作成するメソッド
    def make_list_of_dict_from_mul_list(self, *args) -> List[str]:
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
        dataset = ["normal_prev", "normal_follow", "sas_prev", "sas_follow"]
        preprocess_type = ["spectrum", "spectrogram", "cepstrum"]
        ss_pos = ["bottom", "middle", "top"]
        stride = ["stride_" + str(i) for i in (1, 4, 16, 480, 1024)]
        kernel = ["kernel_" + str(i) for i in (512, 1024)]

        # 辞書の初期値のみをループとは別に作成しておく
        kernel_d = {_kernel: "" for _kernel in kernel}
        list4loop = [stride, ss_pos, preprocess_type, dataset]

    def make_deep_dict(
        self, list4loop: List[Any], first_dict: Dict[str, List[Any]]
    ) -> None:
        for _list4loop in list4loop:
            merged_d = {__list4loop: first_dict for __list4loop in _list4loop}
            first_dict = merged_d.copy()
        # jsonの中身の書き換え
        self.json_dict = merged_d
        # jsonfileへの書き込み
        try:
            with open(self.json_file, "w") as f:
                json.dump(self.json_dict, f, indent=2)
        except Exception:
            print("jsonへの書き込みに失敗しました")
            sys.exit(1)

    def make_model_id_format(self) -> None:
        # 辞書の初期値のみをループとは別に作成しておく
        init_d = {_ss_list: [] for _ss_list in self.ss_list + ["5stage"]}
        list4loop = [
            self.subjects_list,
            self.cleansing_type,
            self.kernel,
            self.stride,
            self.ss_pos,
            self.preprocess_type,
            self.model_type,
            self.dataset,
            self.os_version,
        ]
        self.make_deep_dict(list4loop, init_d)

    def make_pre_processed_id_format(self) -> None:
        # 辞書の初期値のみをループとは別に作成しておく
        kernel_d = {_kernel: "" for _kernel in self.kernel}
        # リストの後ろにあるものほど、後にループ処理がかかるのでJsonの浅い場所のキーとなる
        list4loop = [
            self.stride,
            self.ss_pos,
            self.preprocess_type,
            self.dataset,
            self.os_version,
        ]
        self.make_deep_dict(list4loop, kernel_d)

    # subjects_info.jsonのフォーマット作成
    def make_subjects_info_format(self) -> None:
        # 読み込んだjson形式の末尾に追加する項目
        added_key_d = {_added_key: "" for _added_key in self.added_key}
        merged_d = {__subject: added_key_d for __subject in self.subjects_list}
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
        elif json == "subjects_list":
            # self.make_subjects_list_format()
            print(
                PyColor().RED_FLASH,
                f"{json} is not implemented. Skipping .. ",
                PyColor().END,
            )
            return
        elif json == "subjects_info":
            self.make_subjects_info_format()
        else:
            print(
                PyColor().RED_FLASH,
                f"{json} is not implemented. Skipping .. ",
                PyColor().END,
            )
            return

    # host名の第一キーを返す
    def get_hostkey(self) -> str:
        if self.hostname == "HomeDesktop":
            return "home-pc"
        elif self.hostname == "Spica-2nd":
            return "spica-2nd"
        elif self.hostname == "Castor-2nd":
            return "win_249"
        elif self.hostname == "Spica-4th":
            return "spica-4th"

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
        # "pre_processed_id.json",
        "model_id.json",
        # "my_color.json",
        # "ss.json",
        # "subjects_list.json",
        # "subjects_info.json",
    ]
    for _filenames in filenames:
        jb = JsonBase(_filenames)
        jb.load()
        # フォーマット作成のテスト
        jb.make_format(json=_filenames.split(".")[0])
