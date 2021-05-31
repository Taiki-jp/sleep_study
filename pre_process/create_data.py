# ================================================ #
# *            ライブラリのインポート
# ================================================ #

from random import sample
from matplotlib.pyplot import autoscale, magnitude_spectrum
from record import Record, multipleRecords
from my_setting import *
SetsPath().set()
from scipy import fftpack
from statistics import mean, median, variance, stdev
import numpy as np
from scipy.signal import hamming
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================================================ #
# *     データの加工を行うクラスの作成
# ================================================ #

class CreateData(object):
    """Tanita のデータと睡眠段階のデータを Record に変換する
    """
    def __init__(self):
        pass

    def makeSpectrum(self, tanita_data, psg_data, kernel_size, stride):
        # NOTE : record_lenは公式から簡単に求められる
        record_len = int((len(tanita_data)-kernel_size)/stride)+1
        records = multipleRecords(record_len)
        start_points_list = [i for i in range(0, len(tanita_data)-1024, stride)]
        assert record_len == len(start_points_list)
        
        def _make(start_point, record):
            end_point = start_point+kernel_size
            amped = hamming(len(tanita_data['val'][start_point:end_point])) * tanita_data['val'][start_point:end_point]
            fft = np.fft.fft(amped) / (len(tanita_data['val'][start_point:end_point]) / 2.0)
            fft = 20 * np.log10(np.abs(fft))
            fft = fft[:int(kernel_size/2)]
            record.spectrum = fft
            record.time = tanita_data['time'][int((start_point+end_point)/2)]
        
        def _match(record):
            for counter, psg_time in enumerate(psg_data["time"]):
                if psg_time == record.time:
                    record.ss = psg_data["ss"][counter]
        
        for start_point, record in tqdm(zip(start_points_list, records)):
            _make(start_point=start_point, record=record)
            _match(record=record)
        return records
    
    def makeSpectrogram(self, tanita_data, psg_data, sampleLen=1024, timeLen = 128):
        loopLen = int((len(tanita_data)-sampleLen)/4)  # fft ができる回数
        recordLen = int(loopLen/timeLen)  # record オブジェクトを作る回数（128回のfftに関して1回作る）
        records = multipleRecords(recordLen)
        def _make():
            for i, record in enumerate(tqdm(records)):
                spectroGram = list()
                stop = 0
                for k in range(timeLen):
                    start = k*4+(4*timeLen)*i
                    stop = start+sampleLen
                    if stop > len(tanita_data):
                        print(f'stop index is {stop}. this is out of range tanita data {len(tanita_data)}')
                        break
                    amped = hamming(len(tanita_data['val'][start:stop])) * tanita_data['val'][start:stop]
                    fft = np.fft.fft(amped) / (len(tanita_data['val'][start:stop]) / 2.0)
                    fft = 20 * np.log10(np.abs(fft))
                    fft = list(fft[:int(len(tanita_data['val'][start:stop]) / 2.0)])
                    spectroGram.append(fft)
                record.spectrogram = spectroGram
                if stop > len(tanita_data):
                    break
                record.time = tanita_data['time'][stop]
                def _match():
                    for l, time in enumerate(psg_data['time']):
                        if time == record.time:
                            return psg_data['ss'][l]
                record.ss = _match()
                if record.ss == None:
                    break
            return records
        return _make()
    
    def makeCepstrogram(self, tanita_data, psg_data, sampleLen=1024, timeLen = 128):
        loopLen = int((len(tanita_data)-sampleLen)/4)  # fft ができる回数
        recordLen = int(loopLen/timeLen)  # record オブジェクトを作る回数（128回のfftに関して1回作る）
        records = multipleRecords(recordLen)
        def _make():
            for i, record in enumerate(tqdm(records)):
                spectroGram = list()
                stop = 0
                for k in range(timeLen):
                    start = k*4+(4*timeLen)*i
                    stop = start+sampleLen
                    if stop > len(tanita_data):
                        print(f'stop index is {stop}. this is out of range tanita data {len(tanita_data)}')
                        break
                    # FFT 変換（スペクトラム）
                    amped = hamming(len(tanita_data['val'][start:stop])) * tanita_data['val'][start:stop]
                    fft = np.fft.fft(amped) / (len(tanita_data['val'][start:stop]) / 2.0)
                    fft = np.log10(np.abs(fft))
                    # FFT 変換（ケプストラム）
                    fft = np.fft.fft(fft) / (len(fft) / 2.0)
                    fft = 20 * np.log10(np.abs(fft))
                    # リスト変換
                    fft = list(fft[:int(len(tanita_data['val'][start:stop]) / 2.0)])
                    spectroGram.append(fft)
                record.spectrogram = spectroGram
                if stop > len(tanita_data):
                    break
                record.time = tanita_data['time'][stop]
                def _match():
                    for l, time in enumerate(psg_data['time']):
                        if time == record.time:
                            return psg_data['ss'][l]
                record.ss = _match()
                if record.ss == None:
                    break
            return records
        return _make()

# ================================================ #
# *            試験用メイン関数
# ================================================ #

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from tanita_reader import TanitaReader
    from psg_reader import PsgReader
    from record import Record
    m_createData = CreateData()
    m_tanitaReader = TanitaReader('H_Hayashi')
    m_tanitaReader.readCsv()
    m_psgReader = PsgReader('H_Hayashi')
    m_psgReader.readCsv()
    m_record = Record()
    data = m_tanitaReader.df['val'][0:1024+511]
    spectroGram = m_createData.makeSpectrogram(tanita_data=m_tanitaReader.df, psg_data=m_psgReader.df)
    plt.imshow(m_record.spectrogram)