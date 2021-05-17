# ================================================ #
# *            ライブラリのインポート
# ================================================ #
from pickle import NONE
import random
from file_reader import FileReader

# ================================================ #
#  *                データ作成
# ================================================ #

class LoadSleepData():

    def __init__(self, input_file_name=None):
        """[summary]

        Args:
            input_file_name ([type]): [description]
            test_id ([type]): [TODO : 今後全員のデータを一括で読み込むときは実装してもよい部分（テストデータと訓練データを分けるため）]
        """
        self.FR = FileReader()
        self.input_filename = self.FR.determinFilePath(input_file_name)
    
    def load_data(self, name=None):
        if name:
            return self.FR.loadNormal(name)
        else:
            return self.FR.loadNormal(self.input_filename)
    
    def load_data_all(self):
        records = list()
        for name in self.FR.name_list:
            records.extend(self.FR.loadNormal(self.FR.determinFilePath(name)))
        return records
    
# ================================================ #
#  *       　テストのためのメイン部分
# ================================================ #

if __name__ == "__main__":

    # 確率的に各被験者からデータを取ってくる
    import numpy as np
    from collections import Counter
    import random
    
    o_LoadSleepData = LoadSleepData("H_Li")
    ss_dict = o_LoadSleepData.makeSleepStagesDict()
    tmp = o_LoadSleepData.data[0]
    print(len(tmp))
    y = [record.ss for record in tmp]
    # None を除く
    y = y[:-1]
    Counter(y)
    dict_y = Counter(y)
    # labelsはnr34, nr2, nr1, rem, wakeの順
    labels = [dict_y[i] for i in range(1, 6)]
    labels_all = sum(labels)
    pr_ss = [label/all for label in labels]
    print(pr_ss)
    random.seed(0)
    base = random.random()
    