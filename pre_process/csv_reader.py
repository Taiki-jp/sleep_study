# ================================================ #
# *            ライブラリのインポート
# ================================================ #

from my_setting import FindsDir
import pandas as pd
import sys, os

# ================================================ #
# *     CSV のデータを読み込むクラス作成
# ================================================ #

class CsvReader(FindsDir):
    """CSV ファイル読み込みのための基底クラス

    Args:
        FindsDir ([type]): [description]
    """
    def __init__(self, target_problem, loadDir, personDir, fileName):
        """初期化メソッド

        Args:
            target_problem ([string]]): [project 名を入れる（プロジェクト名とフォルダ名が異なる場合があるため，プロジェクト名に応じてフォルダ名を決定してくれる）]
            loadDir ([string]): [データのフォルダ構成上必要な部分．全てのプロジェクトにおいて構成を同じにすれば，省ける]
            personDir ([string]): [各被験者データを指定する]
            fileName ([string]): [ファイル名を指定する]
        """
        super().__init__(target_problem=target_problem)
        self.rootDirName = self.returnFilePath()
        self.loadDir = loadDir
        self.personDirName = personDir
        self.fileName = fileName
        self.df = None
        
    def readCsv(self):
        filePath = os.path.join(self.rootDirName,
                                self.loadDir,
                                self.personDirName,
                                self.fileName)
        def _read():
            print(f"*** read {filePath} ***")
            self.df = pd.read_csv(filePath)
        return _read()

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    m_csvReader = CsvReader('sleep', 'datas/my_raw_data', 'H_Hayashi', 'signal_after.csv')
    m_csvReader = CsvReader('sleep', 'datas/raw_data', 'H_Hayashi', 'signal_after.csv')
    m_csvReader.readCsv()