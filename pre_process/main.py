import datetime
import os
import random
import sys

from tqdm import tqdm

from data_analysis.py_color import PyColor
from data_analysis.utils import Utils
from pre_process.create_data import CreateData
from pre_process.file_reader import FileReader
from pre_process.pre_processed_id import PreProcessedId
from pre_process.psg_reader import PsgReader
from pre_process.tanita_reader import TanitaReader


def main():
    # ハイパーパラメータの読み込み
    DATA_TYPE = "spectrogram"
    FIT_POS_LIST = ["middle"]
    STRIDE_LIST = [16]
    KERNEL_SIZE_LIST = [128]
    IS_NORMAL = True
    IS_PREVIOUS = False
    VERBOSE = 2

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
                utils = Utils(
                    IS_NORMAL,
                    IS_PREVIOUS,
                    DATA_TYPE,
                    FIT_POS,
                    STRIDE,
                    KERNEL_SIZE,
                    model_type="pass",
                    cleansing_type="",
                )
                CD = CreateData()
                FR = FileReader(
                    IS_NORMAL,
                    IS_PREVIOUS,
                    DATA_TYPE,
                    FIT_POS,
                    STRIDE,
                    KERNEL_SIZE,
                    model_type="pass",
                    cleansing_type="",
                )
                PPI = FR.my_env.ppi
                PPI.dump(is_pre_dump=True)

                target_folders = FR.my_env.set_raw_folder_path(
                    is_normal=IS_NORMAL, is_previous=IS_PREVIOUS
                )
                # To debug efficiently (not to stop the same patient when cause error), sort target_folders randomly
                target_folders = random.sample(
                    target_folders, len(target_folders)
                )

                for target in tqdm(target_folders):
                    _, name = os.path.split(target)
                    tanita = TanitaReader(
                        target, is_previous=IS_PREVIOUS, verbose=VERBOSE
                    )
                    psg = PsgReader(
                        target, is_previous=IS_PREVIOUS, verbose=VERBOSE
                    )
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
                    preprocessing = CD.make_freq_transform(mode=DATA_TYPE)
                    records = preprocessing(
                        tanita.df,
                        psg.df,
                        kernel_size=KERNEL_SIZE,
                        stride=STRIDE,
                        fit_pos=FIT_POS,
                    )
                    utils.dump_with_pickle(
                        records, name, data_type=DATA_TYPE, fit_pos=FIT_POS
                    )

                PPI.dump(value=date_id)


if __name__ == "__main__":
    main()
