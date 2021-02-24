# ================================================ #
# *            ライブラリのインポート
# ================================================ #

from my_setting import *
SetsPath().set()
from csv_reader import CsvReader
import pandas as pd

# ================================================ #
# *タニタのマットセンサのデータを読み込むクラス作成
# ================================================ #

class TanitaReader(CsvReader):
    def __init__(self, 
                 personDir, 
                 fileName='signal_after.csv', 
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
                                  names=('time', 'val'),
                                  usecols=[1, 3],
                                  header=0)
        return _read()

# ================================================ #
# *            試験用メイン関数
# ================================================ #

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from create_data import CreateData
    m_tanitaReader = TanitaReader('H_Li')
    m_tanitaReader.readCsv()
    m_tanitaReader.df