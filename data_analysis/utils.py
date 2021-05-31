import pickle, datetime, os, wandb
from pre_process.file_reader import FileReader
from random import shuffle, choices, random, seed
from pre_process.my_setting import *
SetsPath().set()
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
from tensorflow.keras.datasets import mnist
from load_sleep_data import LoadSleepData
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from collections import Counter
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import numpy

# 便利な関数をまとめたもの    
class Utils():
    
    def __init__(self) -> None:
        self.project_dir = os.environ['sleep']
        self.figure_dir = os.path.join(self.project_dir, "figures")
        self.video_dir = os.path.join(self.project_dir, "videos")
        self.tmp_dir = os.path.join(self.project_dir, "tmps")
        self.analysis_dir = os.path.join(self.project_dir, "analysis")
        self.models_dir = os.path.join(self.project_dir, "models")
        self.id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.fr = FileReader()
        self.name_list = self.fr.sl.name_list
        self.name_dict = self.fr.sl.name_dict
        self.ss_list = self.fr.sl.ss_list
    
    def dump_with_pickle(self, data, fd, file_name):
        
        file_path = os.path.join(fd.returnDataPath(),
                                "pre_processed_data",
                                file_name+'_'+self.id+'.sav')
        pickle.dump(data, open(file_path, 'wb'))
        
    def showSpectrogram(self, *datas, num=4, path = False):
        """正規化後と正規化前の複数を同時にプロットするためのメソッド
        例
        >>> o_utils.showSpectrogram(x_train, tmp)
        このとき x_trian.shape=(batch, row(128), col(512)) となっている（tmp　も同様）
        NOTE : 周波数を行方向に取るために転置をプログラム内で行っている(data[k].T のところ)

        Args:
            num (int, optional): [繰り返したい回数]. Defaults to 4.
            path (bool, optional): [保存場所を指定したいときはパスを渡す．デフォルトではsleep_study/figures/tmp に入る]. Defaults to False.
        """
        fig = plt.figure()
        for i, data in enumerate(datas):
            for k in range(num):
                ax = fig.add_subplot(len(datas), num, 1+k+num*i)
                im = ax.imshow(data[k].T)
                cbar = fig.colorbar(im)
        if path:
            plt.savefig(path)
        else:
            new_path=os.path.join(self.defaultPath, 
                                  self.id+".png")
            plt.savefig(new_path)
        plt.clf()  

    # accuracy, precision, recall, f-measureを計算する
    def compute_properties(self, df):
        """NOTE : df はwandbの形式に合わせて入っているものとする"""
        tp = df[1][0]
        fp = df[0][0]
        tn = df[0][1]
        fn = df[1][1]
        try:
            acc = (tp+tn)/(tp+fp+tn+fn)   #一致率
        except:
            print("accuracyを計算できません", "Noneを返します")
            acc = None
        try:
            pre = (tp)/(tp+fp)  # 予測の適合度
        except:
            print("presicionを計算できません", "Noneを返します")
            pre = None
        try:
            rec = (tp)/(tp+fn)  # 実際のうちの再現度
        except:
            print("recallを計算できません", "Noneを返します")
            rec = None
        f_m = 2*tp/(2*tp+fp+fn)
        return acc, pre, rec, f_m
    
    # 被験者ごとに合わせて評価値を表示する
    def show_properties(self, df_list):
        # df_list はlist型の9x5の構造になっているようにする
        assert type(df_list) == list
        assert df_list.__len__() == 9
        assert df_list[0].__len__() == 5
        returned_dict = dict()
        for _df_list, name in zip(df_list, self.name_list):
            for df, ss in zip(_df_list, self.ss_list):
                dict_key = name+"_"+ss
                assert df.shape == (2,2)
                returned_dict.update({dict_key:self.compute_properties(df)})
        return returned_dict

    # グラフを並べて表示する
    def plot_images(self, images_arr):
        fig, axes = plt.subplots(1, 5, figsize=(20,20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show() 

    # wandbに画像のログを送る
    def save_image2wandb(self, image, dir2 = "confusion_matrix", 
                         fileName = "cm", to_wandb = False):
        sns.heatmap(image, annot = True, cmap = "Blues", fmt = "d")
        path = os.path.join(self.figure_dir, dir2, fileName+"_"+str(self.date_id)+".png")
        plt.savefig(path)
        if to_wandb:
            im_read = plt.imread(path)
            wandb.log({f"{dir2}":[wandb.Image(im_read, caption = f"{fileName}")]})
        plt.clf()
        return

    # attention, imageを並べて表示する
    def simpleImage(self,
                    image_array,
                    row_image_array,
                    file_path, 
                    x_label = None, 
                    y_label = None,
                    title_array = None):
        assert image_array.ndim == 4 and row_image_array.ndim == 3
        for num, image in enumerate(image_array):
            fig = plt.figure(121)
            ax1 = fig.add_subplot(121)
            ax1.axis('off')
            im1 = ax1.imshow(image[:,:,0].T)
            cbar1 = fig.colorbar(im1)
            ax2 = fig.add_subplot(122)
            #ax2.axis('off')  # input の方はメモリを入れる
            im2 = ax2.imshow(row_image_array[num].T, aspect = 'auto')
            cbar2 = fig.colorbar(im2)
            ax1.set_title("Attention")
            ax2.set_title("Input")
            ax2.set_xticks(np.arange(0, 128+1, 32))
            ax2.set_xticklabels(np.arange(0, 32+1, 8))
            ax2.set_yticks(np.arange(0, 512+1, 64))
            ax2.set_yticklabels(np.arange(0, 8+1, 1))
            if x_label:
                ax1.set_xlabel(x_label)
                ax2.set_xlabel(x_label)
            if y_label:
                ax1.set_ylabel(y_label)
                ax2.set_ylabel(y_label)
            plt.tight_layout()
            plt.suptitle(f'conf : {title_array[num].numpy():.0%}', size=10)
            plt.savefig(os.path.join(file_path, f"{num}"))
            plt.clf()

    # ネットワークグラフを可視化する
    def makeNetworkGraph(self, model, dir2 = "test", fileName = "test"):
        path = self.figure_dir + os.path.join(dir2, fileName+".png")
        SVG(data = model_to_dot(model).create(prog = "dot", format = "svg"))

    # パスの存在を確認をして無ければ作成する
    def check_path(self, path):
        if not os.path.exists(path):
            print(f"フォルダが見つからなかったので以下の場所に作成します．\n {path}")
            os.mkdir(path)

    # 存在しないディレクトリを自動で作成する
    def check_path_auto(self, path):
        dir_list = path.split('\\')
        _dir_list = dir_list
        # dir_listの最後にはフォルダではなくファイル名が来ている想定
        file_path = ''
        # FIXME : dir_list の開始地点はインデックスが5以上のTaikiSenjuの下にする
        # NOTE : root_dirを睡眠のディレクトリとすることでその下を確認する
        _, root_dir_name = self.project_dir
        for _ in dir_list:
            if dir_list[0] == root_dir_name:
                # 先頭要素をpop（睡眠のディレクトリまでpop）
                dir_list.pop(0)
                break
            # 先頭要素をpop
            dir_list.pop(0)
        
        # 以下のforループで更新対象の文字列（パス）
        file_path = self.project_dir
        
        # 上のpopで生き残った（睡眠のディレクトリ以下）ディレクトリの存在チェック
        for dir_name in dir_list:
            file_path = os.path.join(file_path, dir_name)
            self.check_path(file_path)

    def make_confusion_matrix(self, y_true, y_pred, using_pandas = False):
        """混合マトリクスを作成するメソッド

        Args:
            x ([array]]): [入力データ]
            y ([array]]): [正解ラベル]
        """
        try:
            assert np.ndim(y_true)==1
        except:
            print("正解データはlogitsで入力してください（one-hotじゃない形で！）")
            sys.exit(1)
        try:
            assert np.ndim(y_pred)==2
        except:
            print("予測ラベルは確率で出力してください")
        y_pred = np.argmax(y_pred, axis=1) 
        
        # ラベルの番号を睡眠段階に読み替える
        # 5段階の睡眠段階の際はこの方法で良い
        # y_trueはtensorflowのオブジェクトで値を変更できないみたいなので、
        # _y_trueを代わりに作成してそっちに入れる
        _y_true = [i for i in range(y_true.shape[0])]
        _y_pred = [i for i in range(y_pred.shape[0])]
        
        for counter, true_label in enumerate(y_true.numpy()):
            if true_label==0:
                _y_true[counter]="nr34"
            elif true_label==1:
                _y_true[counter]="nr2"
            elif true_label==2:
                _y_true[counter]="nr1"
            elif true_label==3:
                _y_true[counter]="rem"
            elif true_label==4:
                _y_true[counter]="wake"
            else:
                print("sleep stage is out of range")
                sys.exit(1)
                
        for counter, pred_label in enumerate(y_pred):
            if pred_label==0:
                _y_pred[counter]="nr34"
            elif pred_label==1:
                _y_pred[counter]="nr2"
            elif pred_label==2:
                _y_pred[counter]="nr1"
            elif pred_label==3:
                _y_pred[counter]="rem"
            elif pred_label==4:
                _y_pred[counter]="wake"
            else:
                print("sleep stage is out of range")
                sys.exit(1)
                
        cm = confusion_matrix(y_true=_y_true, y_pred=_y_pred)
        
        if using_pandas:
            try:
                df = pd.DataFrame(cm,
                                  index = ["wake", "rem", "nr1", "nr2", "nr34"],
                                  columns = ["wake", "rem", "nr1", "nr2", "nr34"])
            except:
                df = pd.DataFrame(cm)
            return cm, df
        return cm, None

    def makeConfusionMatrixFromInput(self, x, y, model, using_pandas = False):
        """混合マトリクスを作成するメソッド

        Args:
            x ([array]]): [入力データ]
            y ([array]]): [正解ラベル]
            model ([tf.keras.Model]]): [NN モデル]
        """
        pred = model.predict(x)
        cm = confusion_matrix(np.argmax(pred, axis = 1), y)
        if using_pandas:
            try:
                df = pd.DataFrame(cm,
                                  index = ["wake", "rem", "nr1", "nr2", "nr34"],
                                  columns = ["wake", "rem", "nr1", "nr2", "nr34"])
            except:
                df = pd.DataFrame(cm)
            return cm, df
        return cm, None

if __name__ == '__main__':
    
    # seedによるランダム性を確認する
    
    # エラーテスト
    instance = PreProcess(project='sleep_study', input_file_name='fft_norm')
    x_train, _ = instance.makeSleepStageSpectrum()
    y_train, _ = instance.makeSleepStage()
    print(Counter(y_train))
    print(x_train.shape)
    
    # Utils.showSpectrogram のテスト
    import my_setting
    import matplotlib.pyplot as plt
    from utils import PreProcess, Utils
    import tensorflow.keras.preprocessing.image as tf_image
    #from my_model import *
    import tensorflow as tf
    # physical_devices = tf.config.list_physical_devices("GPU")
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    my_setting.SetsPath().set()
    o_findsDir = my_setting.FindsDir("sleep")
    o_preProcess = PreProcess(project="sleep_study", input_file_name="H_Li")
    (x_train, y_train) = o_preProcess.loadData(is_split=True)
    #model.summary()
    #hidden = model(x_train)
    #hidden.shape
    tmp = x_train.copy()
    o_preProcess.maxNorm(x_train)
    o_utils = Utils()
    o_utils.showSpectrogram(x_train, tmp)
    
    # 新しいデータ拡張のテスト
    