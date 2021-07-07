from numpy.lib.function_base import hamming
from pre_process.csv_reader import CsvReader
import pandas as pd
import os


class TanitaReader(CsvReader):
    def __init__(
        self,
        personDir,
        fileName="_signal_after.csv",
        target_problem="sleep",
        loadDir="datas/adding_sleep_datas",
    ):
        super().__init__(target_problem, loadDir, personDir, fileName)

    def readCsv(self):
        filePath = os.path.join(
            self.rootDirName, self.loadDir, self.personDirName, self.fileName
        )

        def _read():
            # print(f"*** read {filePath} ***")
            self.df = pd.read_csv(
                filePath,
                #   names=('time', 'val'),
                usecols=["time", "sensor1"],
                header=0,
            )

        return _read()


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
