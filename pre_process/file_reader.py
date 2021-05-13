# ================================================ #
# *            ライブラリのインポート
# ================================================ #
import pickle, sys, os
from subjects_list import SubjectsList
# ================================================ #
# *      pickle ファイル読み込みのためのクラス
# ================================================ #

class FileReader(object):
    def __init__(self):
        self.date = "20210201-055748"  # スペクトログラム
        #self.date = "20210320-011750"  # スペクトラム版
        self.name_list = SubjectsList().nameList
        self.name_dict = {name : name+"_"+self.date+".sav" for name in self.name_list}
        self.dirName = os.path.join(os.environ['sleep'], "datas", "pre_processed_data")

    def determinFilePath(self, alias):
        try:
            return self.name_dict[alias]
        except:
            print(f"In file_reader.py, there isn't {alias} !!")
            sys.exit(1)

    def loadNormal(self, alias, verbose=0):
        path = os.path.join(self.dirName, alias)
        if verbose == 0:
            print(f"*** load {path} ! ***")
        return pickle.load(open(path, 'rb'))

# ================================================ #
#  *       　テストのためのメイン部分
# ================================================ #

if __name__ == '__main__':
    file_reader = FileReader()
    data = file_reader.loadNormal("H_Li")
    pass