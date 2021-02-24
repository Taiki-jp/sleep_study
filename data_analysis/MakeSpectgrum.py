import os
import numpy as np
from matplotlib import pyplot as plt
import loadpickle as lp

# スペクトルを合成する
# とりあえず全部作るか
# ファイル名をIDと睡眠段階によって分ける
# 人によって繰り返し処理
for id in range(9):
    # １秒ずつずらす
    for time in range(len(lp.Normal_data[id])-30):
        # 30秒間のスペクトグラム作成
        fft_array = np.array([])
        for framespace in range(30):
            fft_array=np.append(fft_array, lp.Normal_data[0][time+framespace].spectrum)
        # np.append しただけではデータが1次元に連なるだけなので，整形
        fft_array=fft_array.reshape(30,512)
        # スペクトログラムで縦軸周波数、横軸時間にするためにデータを転置
        fft_array = fft_array.T
        # ここからグラフ描画
        # グラフをオブジェクト指向で作成する。
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        # データをプロットする。
        im = ax1.imshow(fft_array,
                        extent = [0, 30, 0, 512],
                        aspect = 'auto',
                        cmap = 'jet')

        # カラーバーを設定する。カラーバーとは画像の横に表示される色とスペクトル値の対応を表すもの
        cbar = fig.colorbar(im)
        cbar.set_label('SPL [dBA]')

        # 軸設定する。
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Frequency [Hz]')

        # スケールの設定をする。
        ax1.set_xticks(np.arange(0, 30, 5))
        ax1.set_yticks(np.arange(0, 20000, 500))
        ax1.set_xlim(0, 30)
        ax1.set_ylim(0, 512)

        # ファイル名で睡眠段階を区別するためにラベルを作る
        # ３０秒後のPSGを取ることに注意
        # rawdata では 0: N4, 1: N3, 2: N2, 3: N1, 4: REM, 5: WAKE
        SleepStage = str(lp.Normal_data[id][time+30].PSG)

        # figure/spectrum_2d に保存
        # 動作確認のためhogeフォルダに最初は保存
        # dir_name = "figure/spectrum_2d"
        dir_name = "hoge"
        file_name = f"id{id}_ss{SleepStage}_{time}.png"
        path = os.path.join(dir_name,file_name)
        plt.savefig(path)
        # MakeSpectrum.py:24: RuntimeWarning: More than 20 figures have been opened. 
        # Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. 
        # (To control this warning, see the rcParam `figure.max_open_warning`).
        plt.close()


# グラフを表示する。（繰り返し処理の時は省略）
# plt.show()
# plt.close()

# 動画化は以下のコマンド
# ffmpeg -i image%03d.png -pix_fmt yuv420p output.mp4
