import random
import sys
from pre_process.file_reader import FileReader
from data_analysis.utils import Utils
from pre_process.tanita_reader import TanitaReader
from pre_process.psg_reader import PsgReader
from pre_process.create_data import CreateData
from data_analysis.py_color import PyColor
import datetime
import os
from pre_process.json_base import JsonBase


def main():
    # ハイパーパラメータの読み込み
    DATA_TYPE = "spectrum"
    FIT_POS_LIST = ["middle"]
    STRIDE_LIST = [1024]
    KERNEL_SIZE_LIST = [512]
    IS_NORMAL = True
    IS_PREVIOUS = True

    for FIT_POS in FIT_POS_LIST:
        for STRIDE in STRIDE_LIST:
            for KERNEL_SIZE in KERNEL_SIZE_LIST:

                print(
                    PyColor.RED_FLASH,
                    f"fit_pos: {FIT_POS}",
                    f"stride: {STRIDE}",
                    f"kernel: {KERNEL_SIZE}",
                    PyColor.END,
                )

                date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                # オブジェクトの作成
                CD = CreateData()
                FR = FileReader()
                JB = JsonBase("pre_processed_id.json")
                JB.load()
                JB.dump(
                    keys=[
                        DATA_TYPE,
                        FIT_POS,
                        f"stride_{str(STRIDE)}",
                        f"kernel_{str(KERNEL_SIZE)}",
                    ],
                    value=date_id,
                    is_pre_dump=True,
                )
                utils = Utils()

                target_folders = FR.my_env.set_raw_folder_path(
                    is_normal=IS_NORMAL, is_previous=IS_PREVIOUS
                )
                # 実験を効率よくするためにランダムに並べ替える
                target_folders = random.sample(
                    target_folders, len(target_folders)
                )

                for target in target_folders:
                    _, name = os.path.split(target)
                    tanita = TanitaReader(target, is_previous=IS_PREVIOUS)
                    psg = PsgReader(target, is_previous=IS_PREVIOUS)
                    tanita.read_csv()
                    psg.read_csv()
                    # 最初の時間がそろっていることを確認する
                    try:
                        assert datetime.datetime.strptime(
                            tanita.df["time"][0], "%H:%M:%S"
                        ) == datetime.datetime.strptime(
                            psg.df["time"][0], "%H:%M:%S"
                        )
                    except AssertionError:
                        print(
                            PyColor.RED_FLASH,
                            "tanita, psgの最初の時刻がそろっていることを確認してください",
                            PyColor.END,
                        )
                        print(
                            f"tanita: {tanita.df['time'][0]}, psg: {psg.df['time'][0]}"
                        )
                        sys.exit(1)

                    records = CD.make_spectrum(
                        tanita.df,
                        psg.df,
                        kernel_size=KERNEL_SIZE,
                        stride=STRIDE,
                        fit_pos=FIT_POS,
                        target_name=name,
                    )
                    utils.dump_with_pickle(
                        records,
                        name,
                        data_type=DATA_TYPE,
                        fit_pos=FIT_POS,
                        date_id=date_id,
                    )

                # jsonへの書き込み
                JB.dump(
                    keys=[
                        DATA_TYPE,
                        FIT_POS,
                        f"stride_{str(STRIDE)}",
                        f"kernel_{str(KERNEL_SIZE)}",
                    ],
                    value=date_id,
                )


if __name__ == "__main__":
    main()
