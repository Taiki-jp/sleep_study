import sys, os
from pre_processing_base import PreProcessingBase
from scipy import signal
import numpy as np

class FreqencyMethod(PreProcessingBase):
    def __init__(self,
                 method = "fft_raw",
                 spectrumDim = 512,
                 replace = 0,
                 cutoff = 25):
        super().__init__()
        self.method = method
        self.spectrumDim = spectrumDim
        self.replace = replace
        self.cutoff = cutoff
        pass
    
    def calculateLogFft(self, samples, record, index):
        """FFT 変換と対数変換の計算を行うメソッド

        Args:
            samples ([list]): [fft をかけたいデータを入れる]
            record ([record]): [保存する record オブジェクトを渡す]
            index ([int]): [record 形式の何番目のデータかを表す番号]
            
        Returns:
            [fft] : [fft されたデータを ndarray で返す]
        """
        windowedData = signal.hamming(len(samples)) * samples
        fft = np.fft.fft(windowedData)
        # パワースペクトルを取る
        fft = np.abs(fft) ** 2
        # 対数変換
        fft = np.log10(fft) * 10
        return fft
    
