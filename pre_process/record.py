# ================================================ #
# *            ライブラリのインポート
# ================================================ #

from my_setting import *
SetsPath().set()

# ================================================ #
# *     睡眠段階のデータを読み込むクラス作成
# ================================================ #

class Record(object):

    def __init__(self):
        self.time = None
        self.spectrumRaw = None
        self.spectrum = None
        self.spectrogramRaw = None
        self.spectrogram = None
        self.waveletRaw = None
        self.wavelet = None
        self.ss = None

# ================================================ #
# *     record オブジェクトを複数作成するメソッド
# ================================================ #

def multipleRecords(num):
    tmp = []
    for _ in range(num):
        tmp.append(Record())
    return tmp
        

# ================================================ #
# *            試験用メイン関数
# ================================================ #

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    m_record = Record()