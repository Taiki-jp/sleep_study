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

    # subjects_info.jsonのフォーマット作成
    def make_subjects_info_format(self) -> None:
        # 読み込んだjson形式のコピーをとる
        json_prev = self.json_dict.copy()
        # 読み込んだjson形式の末尾に追加する項目
        added_key = ["birth", "sex", "sleeping_time"]
        added_key_d = {_added_key: "" for _added_key in added_key}
        subjects_list = [
            "H_Li",
            "H_Murakami",
            "H_Yamamoto",
            "H_Kumazawa",
            "H_Hayashi",
            "H_Kumazawa_F",
            "H_Takadama",
            "H_Hiromoto",
            "H_Kashiwazaki",
            "sas_5993",
            "sas_30929",
            "sas_33792",
            "sas_35818",
            "sas_35866",
            "sas_35997",
            "sas_36162",
            "sas_36350",
            "sas_36765",
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
            "190000_takadama_aki",
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
            "13724",
            "3325",
            "47328",
            "48253",
            "48797",
            "48814",
            "51443",
            "52274",
            "52345",
            "52470",
            "52819",
            "52868",
            "53224",
            "53354",
            "53420",
            "53670",
            "53776",
            "53776_2",
            "53788",
            "53803",
            "53805",
            "53929",
            "54083",
            "54097",
            "54185",
            "54384",
            "54448",
            "54511",
            "54532",
            "54548",
            "54560",
            "54563",
            "54668",
            "54700",
            "54817",
            "54823",
            "54838",
            "55128",
            "141010_Kashiwazaki_F",
            "141022_Ishii",
            "141212_Saito",
            "141215_Tomura",
            "151125_Umenai",
        ]

        merged_d = {__subject: added_key_d for __subject in subjects_list}
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
        elif json == "subjects_list":
            # self.make_subjects_list_format()
            print("実装まだです．飛ばします")
            return
        elif json == "subjects_info":
            self.make_subjects_info_format()
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
        # "pre_processed_id.json",
        # "model_id.json",
        # "my_color.json",
        # "ss.json",
        # "subjects_list.json",
        "subjects_info.json"
    ]
    for _filenames in filenames:
        jb = JsonBase(_filenames)
        jb.load()

        # フォーマット作成のテスト
        jb.make_format(json=_filenames.split(".")[0])
