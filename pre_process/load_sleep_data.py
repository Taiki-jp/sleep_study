# ================================================ #
# *            ライブラリのインポート
# ================================================ #
import random
from file_reader import FileReader

# ================================================ #
#  *                データ作成
# ================================================ #

class LoadSleepData():

    def __init__(self, 
                 input_file_name):
        """[summary]

        Args:
            input_file_name ([type]): [description]
            test_id ([type]): [TODO : 今後全員のデータを一括で読み込むときは実装してもよい部分（テストデータと訓練データを分けるため）]
        """
        self.m_fileReader = FileReader()
        self.inputFileName = self.m_fileReader.determinFilePath(input_file_name)
        self.data = self.m_fileReader.loadNormal(self.inputFileName)
            
    def makeSleepStagesDict(self):
        ss_label = ["nr4", "nr3", "nr2", "nr1", "rem", "wake"]
        sleep_stages_dict = dict()
        for i in range(6):
            tmp_list = [record for record in self.data[0] if record.ss == i]
            tmp_dict = {ss_label[i] : tmp_list}
            sleep_stages_dict.update(**tmp_dict)
        return sleep_stages_dict
    
    def makeTestDataEachStage(self):
        return self.m_changeLabel.makeTestList()
    
    def makeTrainData(self):
        trainData = self.nr1Data+self.nr2Data+self.nr34Data+self.remData+self.wakeData
        random.shuffle(trainData)
        return trainData
    
    def makeTestData(self):
        return self.nr1DataTest+self.nr2DataTest+self.nr34DataTest+self.remDataTest+self.wakeDataTest 
    
    def storchasticSampling(self, seed, before_decreese = True):
        pass
# ================================================ #
#  *       　テストのためのメイン部分
# ================================================ #

if __name__ == "__main__":
    
    # main で実行するときはデータを確認する
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter
    record_2d = list()
    # TODO : nameDict は utils のname_dict から取ってくる
    for name in nameDict.keys():
        m_loadSleepData = LoadSleepData(input_file_name=name)
        record_2d.extend(m_loadSleepData.data[0])
        del m_loadSleepData
        # record_2d は（被験者数, データ数）になっている
        # データ数は順に 794, 557, 806, 674, 568, 393, 547, 610, 659

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
    