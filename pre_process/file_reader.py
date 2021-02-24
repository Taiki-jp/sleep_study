# ================================================ #
# *            ライブラリのインポート
# ================================================ #

import pickle, sys, os

# ================================================ #
# *      pickle ファイル読み込みのためのクラス
# ================================================ #

class FileReader(object):
    def __init__(self):
        self.date = "20210201-055748"
        self.name_list = \
            [
            "H_Li",
            "H_Hayashi",
            "H_Hiromoto",
            "H_Kashiwazaki",
            "H_Kumazawa",
            "H_Kumazawa_F",
            "H_Murakami",
            "H_Takadama",
            "H_Yamamoto",
            ]
        self.name_dict = {name : name+"_"+self.date+".sav" for name in self.name_list}
        self.dirName = os.path.join(os.environ['sleep'], "datas", "pre_processed_data")

    def determinFilePath(self, alias):
        try:
            return self.name_dict[alias]
        except:
            print(f"In file_reader.py, there isn't {alias} !!")
            sys.exit(1)

    def loadNormal(self, alias):
        path = os.path.join(self.dirName, alias)
        print(f"*** load {path} ! ***")
        return pickle.load(open(path, 'rb'))

# ================================================ #
#  *       　テストのためのメイン部分
# ================================================ #

if __name__ == '__main__':
    pass