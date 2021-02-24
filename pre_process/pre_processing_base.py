import sys, os
from tanita_reader import TanitaReader
from psg_reader import PsgReader

def PreProcessingBase(object):
    def __init__(self, 
                 windowLen = 64,
                 hz = 16):
        self.windowLen = windowLen
        self.spectrumDim = spectrumDim
        self.hz = hz
        pass
                
    def getSamples(self, 
                psgDateTime, 
                tanitaDf,
                startingPoint = 0):
        """psg の時刻に合うデータ（できれば 16 個）を取ってくる関数． \
            実際は数の制限をしていないので何個とってくるかは分かんない

        Args:
            psgDateTime ([datetime64]): [psg の時刻]
            tanitaDf ([dateframe]): [tanita のデータフレーム]
            startingPoint ([int]): [繰り返しのポイント]

        Returns:
            [list]: [tanita のセンサ値を取ってくる]
        """
        samples = []
        for tanitaDateTime in tanitaDf:
            if psgDateTime == tanitaDateTime:
                samples.append(tanitaDf["sensor1"][startingPoint])
                
        # どれだけずれるか確認したら消してok
        if len(samples) < self.hz:
            print(f"Got {len(samples)} data")
        elif len(samples) > self.hz:
            print(f"Got {len(samples)} data")
        return samples
    
    def mergeSamples(self,
                      psgDataTime,
                      tanitaDf):
        """ひとつの睡眠段階を決定するためのサンプルを取るメソッド．\
            getSamples で取ったリストをまとめる処理を行う
        
        Args:
            psgDateTime ([psgDatetime]): [psg の時刻]
            tanitaDf ([tanitaDf]): [tanita のデータフレーム]

        Returns:
            [list]: [tanita のセンサ値 16 個を取ってくる]
        
        """
        mergedSamples = []
        for _ in range(self.windowLen):
            samples = self.getSamples(psgDataTime, tanitaDf)
            mergedSamples.extend(samples)
        return mergedSamples
    
    
        
