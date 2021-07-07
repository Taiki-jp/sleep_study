import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def makeLongSpectroGram(data, hour):
    # TODO : 日付を超えるものに関しては使えない
    # TODO : グラフに時間を与えたければこの関数内でrecordを返す必要がある
    import datetime

    start_time = data[0].time
    start_time = datetime.datetime.strptime(start_time, "%H:%M:%S")
    if type(hour) == int:
        end_time = start_time + datetime.timedelta(hours=hour)
    elif hour == "all" or hour == "A":
        end_time = data[-1].time
        end_time = datetime.datetime.strptime(end_time, "%H:%M:%S")
    else:
        print(f'hour is integer or "all". but given {type(hour)}')
        sys.exit(1)
    del start_time
    target_list = [
        record
        for record in data
        if datetime.datetime.strptime(record.time, "%H:%M:%S") < end_time
    ]
    spectrogram_target = [record.spectrogram for record in target_list]
    ss_target = [record.ss for record in target_list]
    time_target = [record.time for record in target_list]
    tmp = spectrogram_target[0]
    for target in spectrogram_target[1:]:
        tmp = np.vstack((tmp, target))
    return tmp, ss_target, time_target


def show2dImage(x_label, y_label, title, path=None, term=10, **datas):
    figure = plt.figure(figsize=(12, 4))
    ax = figure.add_subplot()
    # drow color map
    try:
        im = ax.imshow(datas["spectrogram"], aspect=60)
        figure.colorbar(im)
    except Exception:
        print("スペクトログラムのデータがセットされていないので描画しません")
    # plot sleep stage
    try:
        _, col = datas["spectrogram"].shape
        x = np.arange(0, col, 128)
        # よさげな場所に描画するために400や10を入れている
        ax.plot(x, 64 * 3 + 64 * datas["ss"], linewidth=3, color="red")
    except Exception:
        print("睡眠段階のデータがセットされていないので描画しません")
    # set title, labels, ticks
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # set axis precicely
    try:
        _, col = datas["spectrogram"].shape
        show_time_label_term = term
        x = np.arange(0, col, 128)[::show_time_label_term]
        ax.set_xticks(x)
        # plt.xticks(rotation=90)
        ax.set_xticklabels(datas["time"][::show_time_label_term])

    except Exception:
        print("時間がセットされていないので描画しません")
    ax.set_yticks(np.arange(0, 512 + 1, 64))
    ax.set_yticklabels(np.arange(0, 8 + 1, 1))
    if path:
        plt.savefig(path)
    else:
        plt.savefig(
            os.path.join(os.environ["SLEEP"], "figures", "tmp", "tmp.png")
        )


def inverseSleepStage(ss_list):
    tmp = list()
    for ss in ss_list:
        if ss == 0:
            tmp.append(5)
        elif ss == 1:
            tmp.append(4)
        elif ss == 2:
            tmp.append(3)
        elif ss == 3:
            tmp.append(2)
        elif ss == 4:
            tmp.append(1)
        elif ss == 5:
            tmp.append(0)
        else:
            print("例外発生")
            sys.exit(1)
    return tmp


# ================================================ #
# *            試験用メイン関数
# ================================================ #

if __name__ == "__main__":

    # 指定時間スケールのスペクトログラム作成関数
    from pre_process.load_sleep_data import LoadSleepData
    from data_analysis.utils import Utils

    name_dict = Utils().name_dict
    for name in name_dict:
        o_loadSleepData = LoadSleepData(input_file_name=name)
        records = o_loadSleepData.data[0]
        print(len(records))  # 794
        print("start : ", records[0].time, "stop : ", records[-1].time)
        hour = "all"
        (spectrogram, ss_list, time_list) = makeLongSpectroGram(
            data=records, hour=hour
        )
        # >> spectrogram.shape = (14464, 512)
        # : len(ss_list) = 113 : len(time_list) = 113
        spectrogram = spectrogram.T
        # >> spectrogram.shape = (512, 14464)
        ss_list = inverseSleepStage(ss_list=ss_list)
        # >> Counter({3: 36, 2: 392, 1: 155, 4: 165, 5: 40})
        # >> Counter({2: 36, 3: 392, 4: 155, 1: 165, 0: 40})
        file_path = os.path.join(
            os.environ["SLEEP"], "figures", "tmp", name + "_ceps" + ".png"
        )
        show2dImage(
            x_label="Time",
            y_label="Frequency",
            title=f"{name} Cepstrogram",
            term=90,
            path=file_path,
            spectrogram=spectrogram,
            ss=np.array(ss_list),  # 和を計算するために list のままだとダメなのでndarrayに変換
            time=time_list,
        )
