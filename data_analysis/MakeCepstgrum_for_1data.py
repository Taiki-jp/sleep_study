# H_Li 用のデータ（データが大きすぎると面倒なので全体の傾向をつかむまではこのファイルを基に分析する
# X Launcher を立ち上げていないと plt.figure() でエラーが出るので注意
# デフォルトでは、matplotlibはTK GUIツールキットを使用します。
# ツールキット(つまり、ファイルまたは文字列)を使用せずに画像をレンダリングするとき、
# matplotlibは表示されないウィンドウをインスタンス化し、あらゆる種類の問題を引き起こします。
# それを避けるために、Aggバックエンドを使用する必要があります。
import os
import numpy as np
import matplotlib
# uncomment matplotlib.use('Agg') if you want to execute interactively.
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import loadpickle as lp
from tqdm import tqdm
# スペクトルを合成する
# とりあえず全部作るか
# ファイル名を睡眠段階によって分ける
# １秒ずつずらす
for time in tqdm(range(len(lp.Normal_data[0])-30)):
    # 30秒間のスペクトグラム作成
    fft_array = np.array([])
    for framespace in range(30):
        # 逆順で入れないと周波数が逆になっている
        fft_array=np.append(fft_array, lp.Normal_data[0][time+framespace].cepstrum[::-1])
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
    ax1.set_yticks(np.arange(0, 600, 100))
    ax1.set_xlim(0, 30)
    ax1.set_ylim(0, 512)
    # ファイル名で睡眠段階を区別するためにラベルを作る
    # ３０秒後のPSGを取ることに注意
    # rawdata では 0: N4, 1: N3, 2: N2, 3: N1, 4: REM, 5: WAKE
    SleepStage = str(lp.Normal_data[0][time+30].PSG)
    # figure/cepstrum_2d に保存
    # 動作確認のためhogeフォルダに最初は保存
    dir_name = "figure/SpectgrumWithColorbarForH_Li"
    # dir_name = "hoge"
    file_name = f"ss{SleepStage}_{time}.png"
    path = os.path.join(dir_name,file_name)
    plt.savefig(path)
    plt.close()

# 動画化は以下のコマンド（うまく動かない時があるから確認），連番の時はこっちの方が便利
# ffmpeg -start_number 90 -i ss1_%d.png -pix_fmt yuv420p ss1_90.mp4
# まとめて * の glob を使って動画化
#  ffmpeg -pattern_type glob -i 'ss1/*.png' -pix_fmt yuv420p movie/ss1All.mp4
# RuntimeError: main thread is not in main loop
# Tcl_AsyncDelete: async handler deleted by the wrong thread
# 理由：tkinter がスレッドセーフじゃないことが原因みたい
