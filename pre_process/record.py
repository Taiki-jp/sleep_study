from dataclasses import dataclass
import numpy as np

@dataclass
class Record(object):
    time : str = ""
    spectrum_raw : np.ndarray = np.array([]) 
    spectrum : np.ndarray = np.array([])
    spectrogram_raw : np.ndarray = np.array([])
    spectrogram : np.ndarray = np.array([])
    wavelet_raw : np.ndarray = np.array([])
    wavelet : np.ndarray = np.array([])
    ss : int = None

def multipleRecords(num):
    tmp = []
    for _ in range(num):
        tmp.append(Record())
    return tmp
        

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from data_analysis.py_color import PyColor
    record = Record()
    for key, val in record.__dict__.items():
        print(PyColor.GREEN,
              f"key : {key}, ",
              f"val : {val}",
              PyColor.END)