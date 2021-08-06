from numpy.lib.function_base import hamming
from pre_process.csv_reader import CsvReader
import pandas as pd
import os


class TanitaReader(CsvReader):
    def __init__(
        self,
        person_dir: str,
        file_name: str = "signal_after.csv",
        verbose: int = 0,
        is_previous: bool = True,
        columns: list = [1, 3],
        names: list = ["time", "ss"],
    ):
        super().__init__(
            person_dir=person_dir,
            file_name=file_name,
            verbose=verbose,
            is_previous=is_previous,
            columns=columns,
            names=names,
        )
        if not is_previous:
            # ファイル名の先頭に_を入れる
            self.file_name = "_" + file_name
            # 2列目と3列目をとってくる
            self.columns = [1, 2]


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    m_tanitaReader = TanitaReader("H_Li")
    m_tanitaReader.readCsv()
    m_tanitaReader.df

    # 全ての時刻のデータ
    all_datas = m_tanitaReader.df["val"].to_numpy()

    # 最初の1024このデータ
    target_datas = all_datas[1024:2048]

    # 窓関数
    windowing = hamming(1024) * target_datas

    # FFT変換
    fft = np.fft.fft(windowing) / 1024 * 2

    # パワースペクトル
    fft_abs = np.abs(fft) ** 2

    # 対数スケール
    fft_abs_log_scale = 10 * np.log10(fft_abs)

    # グラフ描画の関数
    def plot_images(array, filename):
        import seaborn as sns

        sns.set()
        fig, ax = plt.subplots()
        ax.plot(array)
        # ax.axis('off')
        plt.tight_layout()
        path = os.path.join(os.environ["sleep"], "tmps", f"{filename}.png")
        plt.savefig(path)

    def makeSpectrum(self, data, record=False):
        """(windowing) x (FFT) x (normalize) x (log scale)"""
        amped = hamming(len(data)) * data
        fft = np.fft.fft(amped, norm="ortho") / (len(data) / 2.0)
        fft = 20 * np.log10(np.abs(fft))
        if record:
            record.spectrum = list(fft[: int(len(data) / 2)])
            return
        return fft
