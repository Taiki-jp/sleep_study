import os
from glob import glob
import pandas as pd


# TODO : 睡眠データのリスト取得
def get_data_per_day(start_day=20210303) -> list:

    print(f"*** {start_day} 以降のデータを確認します ***")

    target_dir = os.environ["GAMMA_SAS"]
    saved_data_list = glob(target_dir + r"/*")
    data_len = len(saved_data_list)

    # 返すリスト（指定した日付より新しいもの）
    return_dir = list()

    for counter, saved_data in zip(range(data_len), saved_data_list):
        # 最後のフォルダ名だけを得る
        _, date_num = os.path.split(saved_data)

        # int型にキャストできないもの（日付フォルダ以外）もあるかもしれないのでtry-except処理
        try:
            date_num = int(date_num)
        # flake8のE722を解消するためにexcept文にエラークラスを明示
        except TypeError:
            print(f"{date_num} はint型にキャストできません")
            saved_data_list.pop(counter)
            continue

        # 日付が指定したstart_dayよりも新しい（値が大きい）ければ，返すリストに追加
        if start_day < date_num:
            return_dir.append(saved_data)

    return return_dir


# TODO : フォルダのサイズをチェック（サイズ以外でもチェックできるように）
def check_data_size(path: str, base_size=1.0e6) -> bool:
    data_size = get_dir_size(path)
    if data_size > base_size:
        return True
    else:
        return False


# フォルダのサイズを取得する関数（単位：バイト）
def get_dir_size(path: list) -> int:
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


def write2csv(emfit_list: list, tanita_list: list, path=None) -> None:
    df_emfit = pd.DataFrame(emfit_list, columns=["emfit"])
    df_tanita = pd.DataFrame(tanita_list, columns=["tanita"])

    # もしパスが指定されていたら指定されたパスに保存
    if bool(path):
        save_path = path
    else:
        save_path = os.getcwd()

    # 書き込み
    df_emfit.to_csv(save_path + r"/emfit.csv")
    df_tanita.to_csv(save_path + r"/tanita.csv")
