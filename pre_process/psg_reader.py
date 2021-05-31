# ================================================ #
# *            ライブラリのインポート
# ================================================ #

from my_setting import *
SetsPath().set()
from csv_reader import CsvReader
import pandas as pd

# ================================================ #
# *     睡眠段階のデータを読み込むクラス作成
# ================================================ #

class PsgReader(CsvReader):
    def __init__(self, 
                 personDir,
                 fileName='sleepStage.csv',
                 target_problem='sleep',
                 loadDir='datas/raw_data'):
        super().__init__(target_problem, loadDir, personDir, fileName)

    def readCsv(self):
        filePath = os.path.join(self.rootDirName,
                                self.loadDir,
                                self.personDirName,
                                self.fileName)
        def _read():
            #print(f"*** read {filePath} ***")
            self.df = pd.read_csv(filePath, 
                                  names=('time', 'ss'),
                                  usecols=[0, 2],
                                  header=0)
        return _read()
    
# ================================================ #
# *            試験用メイン関数
# ================================================ #

# main として実行していない時は ファイル名（この場合は csv_reader）として __name__ に入っている
if __name__ == '__main__':
    m_psgReader = PsgReader('H_Hayashi')
    m_psgReader.readCsv()
    m_psgReader.df