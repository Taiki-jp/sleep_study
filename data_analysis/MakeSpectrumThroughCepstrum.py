import loadpickle as lp
from scipy import fftpack
import pandas as pd
import numpy as np
from tqdm import tqdm
# 初期化
matrix = np.array([])
for i in tqdm(range(len(lp.Normal_data[0]))):
    x = lp.Normal_data[0][i].spectrum
    # 逆フーリエ変換（ケプストラム） 使えるのは 4 Hz, 256 sample
    ceps_db = np.real(fftpack.ifft(x))
    # ローパスリフタ
    index = 25
    ceps_db[index:len(ceps_db)-index] = 0
    # フーリエ変換　周波数領域に戻す　使えるのは 2 Hz, 128 sample
    ceps_db_low = fftpack.fft(ceps_db)
    # 25439 * 128 の行列に保存してpandasで保存する
    matrix = np.append(matrix, ceps_db_low)
# 保存
df = pd.DataFrame(matrix.reshape(25439, 512))
df.to_csv('csv/spectrum_H_Li.csv')