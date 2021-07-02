from pre_process.csv_reader import CsvReader
import pandas as pd
import os

# ================================================ #
# *     睡眠段階のデータを読み込むクラス作成
# ================================================ #

class PsgReader(CsvReader):
    def __init__(self, 
                 personDir,
                 fileName='_sleepStage.csv',
                 target_problem='sleep',
                 loadDir='datas/adding_sleep_datas'):
        super().__init__(target_problem, loadDir, personDir, fileName)

    def readCsv(self):
        filePath = os.path.join(self.rootDirName,
                                self.loadDir,
                                self.personDirName,
                                self.fileName)
        def _read():
            #print(f"*** read {filePath} ***")
            self.df = pd.read_csv(filePath, 
                                #   names=('time', 'ss'),
                                  usecols=["time", "ss"],
                                  header=0)
        return _read()
    
if __name__ == '__main__':
    m_psgReader = PsgReader('H_Hayashi')
    m_psgReader.readCsv()
    m_psgReader.df