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

def multipleRecords(num):
    tmp = []
    for _ in range(num):
        tmp.append(Record())
    return tmp
        

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    record = Record()
    print(record.time)