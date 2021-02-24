# ================================================ #
# *            ライブラリのインポート
# ================================================ #

from collections import Counter
import sys, random

# ================================================ #
# *      5 クラス分類のための前処理クラス
# ================================================ #

class ChangeLabel(object):
    
    def __init__(self, record, testId):
        """初期化メソッド

        Args:
            record ([record object]): [record object のリスト型で入れる]
            testId (int, optional): [テストデータとする人物の id]. Defaults to 1. \n
            n34 を含むもの（テストデータとして選ぶことが妥当な被験者は \n
            id）Li(0), Murakami(1), Kumazawa(3), Hayashi(4), Takadama(6) のようになっている
        """
        self.nr4LabelFrom = 0
        self.nr3LabelFrom = 1
        self.nr2LabelFrom = 2
        self.nr1LabelFrom = 3
        self.rLabelFrom = 4
        self.wLabelFrom = 5
        self.wLabelTo = 0
        self.rLabelTo = 1
        self.nr1LabelTo = 2
        self.nr2LabelTo = 3
        self.nr34LabelTo = 4
        self.nr1DataList = []
        self.nr1DataList = []
        self.nr2DataList = []
        self.nr34DataList = []
        self.rDataList = []
        self.wDataList = []
        self.nr1DataListInd = []
        self.nr1DataListInd = []
        self.nr2DataListInd = []
        self.nr34DataListInd = []
        self.rDataListInd = []
        self.wDataListInd = []
        self.testId = testId
        self.nameList = [name for name in range(len(record))]
        self.record = record
        pass
    
    def initializeList(self):
        """list を初期化するメソッド． \n
        被験者のループが終わった際に呼び出されることを想定して作成．
        """
        self.nr1DataListInd = []
        self.nr2DataListInd = []
        self.nr34DataListInd = []
        self.rDataListInd = []
        self.wDataListInd = []
        return
    
    def popTestData(self):
        """オブジェクト生成時に指定した id を基にテストデータとなる被験者をリストから外す
        """
        self.nameList.pop(self.testId)
        print(f"test data id is {self.testId}!!")
        return
    
    def extendList(self, isSpectrum = True):
        """各被験者の睡眠段階を他のリストに格納するためのメソッド． \n
        各被験者のループのたびにリストは初期化されるので，別で保存する必要があった．
        """
        #cutsize = self.defineTrainSize(isSpectrum = isSpectrum)
        #nr1 = random.sample(self.nr1DataListInd, cutsize)
        #nr2 = random.sample(self.nr2DataListInd, cutsize)
        #rem = random.sample(self.rDataListInd, cutsize)
        #wake = random.sample(self.wDataListInd, cutsize)

        # 全部入れる
        nr1 = self.nr1DataListInd
        nr2 = self.nr2DataListInd
        rem = self.rDataListInd
        wake = self.wDataListInd

        self.nr1DataList.extend(nr1)
        self.nr2DataList.extend(nr2)
        self.nr34DataList.extend(self.nr34DataListInd)
        self.rDataList.extend(rem)
        self.wDataList.extend(wake)
    
    def labelChecker(self, data):
        """PSG のラベルを変更するメソッド
        - 元のラベルのままだと睡眠段階が深くなるにつれて数値が小さくなる． \n
        - これを睡眠段階が深くなるにつれて高くなるように変更するメソッドである． \n
        - これは混合マトリクス表示のために実装を行ったが，
        数字が小さい方から表示してくれていない現状としてはいまいち意味がない \n
        - さらに，6 段階の元々のものを 5 段階に変更する効果もある

        Args:
            data ([record]): [各被験者の各時刻の record が入る]
        """
        if data.ss == self.nr4LabelFrom or data.ss == self.nr3LabelFrom:
            data.ss = self.nr34LabelTo
            self.nr34DataListInd.append(data)
            
        elif data.ss == self.nr2LabelFrom:
            data.ss = self.nr2LabelTo
            self.nr2DataListInd.append(data)
            
        elif data.ss == self.nr1LabelFrom:
            data.ss = self.nr1LabelTo
            self.nr1DataListInd.append(data)
            
        elif data.ss == self.rLabelFrom:
            data.ss = self.rLabelTo
            self.rDataListInd.append(data)
            
        elif data.ss == self.wLabelFrom:
            data.ss = self.wLabelTo
            self.wDataListInd.append(data)

        else:
            print(f"unknown sleep stage {data.ss} came !!")
            # sys.exit(1)
    
    def _change(self):
        for data in self.record[0]:  # NOTE : 各被験者ごとのデータの時は毎回0番目を見る
            if data.ss == self.nr1LabelFrom:
                data.ss = self.nr1LabelTo
            elif data.ss == self.nr2LabelFrom:
                data.ss = self.nr2LabelTo
            elif data.ss == self.nr3LabelFrom or data.ss == self.nr4LabelFrom:
                data.ss = self.nr34LabelTo
            elif data.ss == self.rLabelFrom:
                data.ss = self.rLabelTo
            elif data.ss == self.wLabelFrom:
                data.ss = self.wLabelTo

    def makeTrainList(self):
        self._change()
        return random.shuffle(self.record[0])

    def fixLabel(self):
        ssList = [record.ss for record in self.record[0][:]]
        ssDict = Counter(ssList)
        print(ssDict)
        
# ================================================ #
# *            試験用メイン関数
# ================================================ #
# Counter({1: 314, 2: 285, 4: 154, 5: 23, 3: 18})
# 
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter
    from load_sleep_data import LoadSleepData  # TODO : 循環インポートになっているためできない
    # main で実行するときはデータを確認する
    inputFileName = input("入力データを入れてください（load_sleep_data をメイン関数として実行した際に表示）\n")
    m_loadSleepData = LoadSleepData(input_file_name=inputFileName, test_id=0)
    m_changeLabel = ChangeLabel(m_loadSleepData.data, testId=None)
    m_changeLabel.fixLabel()

