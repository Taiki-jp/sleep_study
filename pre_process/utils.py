# ================================================ #
# *            ライブラリのインポート
# ================================================ #
from random import shuffle, choices, random, seed
from file_reader import FileReader
from my_setting import *
SetsPath().set()
import os, wandb
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
import pickle
import datetime

# ================================================ #
# *            主に画像の前処理クラス
# ================================================ #

class PreProcess():
    
    def __init__(self, project, input_file_name=None):
        # project は project のディレクトリとなることに注意
        self.projectDir = os.environ['sleep']
        self.figureDir = os.path.join(self.projectDir, "figures")
        self.videoDir = os.path.join(self.projectDir, "videos")
        self.tmpDir = os.path.join(self.projectDir, "tmps")
        self.analysisdir = os.path.join(self.projectDir, "analysis")
        self.modelsDir = os.path.join(self.projectDir, "models")
        self.inputFileName = input_file_name
        self.file_dict = None
        self.name_dict = Utils().name_dict
        self.test_data_for_wandb = None
        seed(0)
        
    def checkRandom(self):
        tmp = [randint(0, 10) for _ in range(10)]
        print(tmp)
    
    def list2Spectrum(self, list_data):
        return np.array([data.spectrum for data in list_data])
    
    def list2SS(self, list_data):
        return np.array([k.ss for k in list_data])
    
    def list2Spectrogram(self, list_data):
        return np.array([data.spectrogram for data in list_data])
    
    def loadData(self, is_split, is_auto_loop=False, is_auto_loop_name=False):
        
        assert type(self.inputFileName) == str or type(self.inputFileName) == dict
        
        if type(self.inputFileName) == str:
            records = LoadSleepData(self.inputFileName).data[0][:]
            return self.list2Spectrogram(records), self.list2SS(records)
        elif type(self.inputFileName) == dict:
            
            def _make(test_data):
                records_train = list()
                records_test = list()
                for name in self.name_dict.keys():
                    _loadSleepData = LoadSleepData(name)
                    if name == test_data:
                        records_test.extend(_loadSleepData.data[0])
                    elif name != test_data:
                        records_train.extend(_loadSleepData.data[0])
                    del _loadSleepData
                return (records_train, records_test)
            
            if is_split:
                # 被験者データに対して自動処理を行いたいときはTrueにする
                if is_auto_loop:
                    assert is_auto_loop_name != False
                    # print("被験者データに対して自動でループ処理を行います")
                    test_data = is_auto_loop_name
                    print(f"テストデータは{test_data}です")
                    self.test_data_for_wandb = test_data
                    (train, test) = _make(test_data=test_data)
                    
                else:
                    print(self.name_dict.keys())
                    test_data = input("以上からテストデータを選んでください：")
                    #test_data = "H_Li"
                    print(f"テストデータは{self.name_dict[test_data]}です")
                    self.test_data_for_wandb = test_data
                    (train, test) = _make(test_data=test_data) 
                    
                return (train, test)     
    
    def makeDataSet(self, train, test,
                    is_split = False, 
                    is_set_data_size = True, 
                    target_ss = None, 
                    is_storchastic = True, 
                    is_shuffle = True,
                    is_multiply = True,
                    mul_num = None):
            
            if is_split:
                # ここまでの操作を別の関数で行う方が良い
                if is_set_data_size:
                    assert target_ss != None
                    tmporary_sleep_stage = [record.ss for record in train]
                    #print("訓練データのサイズをそろえる前のもの", Counter(tmporary_sleep_stage))
                    # ss_dict_train は各睡眠段階の比率（数）を表す
                    # target_ss も返して大丈夫？
                    ss_dict_train = self.setDataSize(train, target_ss=target_ss, isStorchastic=is_storchastic)
                    #ss_dict_test = self.setDataSize(test, target_ss=target_ss, isStorchastic=True)  # NOTE : テストデータに対しては確率的サンプリングを行う
                    ss_dict_test = Counter([record.ss for record in test])
                    #print("データサイズをそろえたもの（訓練）", ss_dict_train)
                    #print("データサイズをそろえたもの（テスト）", ss_dict_test)
                    def _storchastic_sampling(data, target_records, is_test):
                        tmp = list()
                        
                        def _splitEachSleepStage():
                            each_stage_list = list()
                            
                            for ss in data.keys():
                                # record は順番通りにソートされていない
                                # 多分train自体がソートされたものではないから
                                # data.keys()の順番が5, 2, 4, 3, 1になっているから？
                                # でもその順番に追加されているわけではないから違いそう
                                # なんか最初に4が選択されていた（辞書なので順番がないからそういうものと思う）
                                each_stage_list.append([record for record in target_records if record.ss == ss])
                            return each_stage_list
                        # record を各睡眠段階ごとにわけているもの（長さ５とは限らない（nr3がない場合））
                        ss_list = _splitEachSleepStage()
                        for records in ss_list:
                            if len(records) == 0:
                                print("睡眠段階が存在しないため飛ばします")
                                continue
                            # 各睡眠段階がある間はその睡眠段階に対してランダムサンプリングを行う
                            def _sample(target_records):
                                # 睡眠段階のラベルを知るために必要
                                if is_test:
                                    ss = target_records[0].ss
                                    l = choices(target_records, k=data[ss])
                                else:
                                    ss = target_records[0].ss
                                    if ss != target_ss:
                                        try:
                                            # target_recordsからdata[ss]個取ってくる感じ
                                            # >>> len(target_records)
                                            # >>> 1019（各睡眠段階のrecordオブジェクトが入っている
                                            # >>> data[4]
                                            # >>> 睡眠段階が4のもの（REM）の数を返す（正確にはCounterの辞書の4に対応するvalueのこと）
                                            if is_multiply:
                                                # l = sample(target_records, 4*data[ss])
                                                l = choices(target_records, k=400)
                                            else:
                                                if mul_num != None:
                                                    l = choices(target_records, k=data[ss]*mul_num)
                                                else:
                                                    l = choices(target_records, k=data[ss])
                                        except:
                                            if is_multiply:
                                                #print("データを複製する必要があるので、random.choices()を用います")
                                                l = choices(target_records, k=400)
                                            else:
                                                if mul_num != None:
                                                    l = choices(target_records, k=data[ss]*mul_num)
                                                else:
                                                    l = choices(target_records, k=data[ss])
                                    elif ss == target_ss:
                                        #l = choices(target_records, k=400)
                                        l = choices(target_records, k=data[ss])
                                return l
                            tmp.extend(_sample(target_records=records))
                        return tmp
                    
                    train = _storchastic_sampling(data=ss_dict_train, target_records = train, is_test = False)
                    # trainは4倍に複製できてる
                    test = _storchastic_sampling(data=ss_dict_test, target_records = test, is_test = True)
                    # testは複製していない
                    if is_shuffle:
                        #print("シャッフルします")
                        #print("最初の先頭", train[0])
                        shuffle(train)
                        #print("シャッフル後の先頭", train[0])
                return (self.list2Spectrogram(train), self.list2SS(train)), (self.list2Spectrogram(test), self.list2SS(test))

            else:
                print("テストデータは用意しません（テストの際は交差検証を試してください）")
                train, _ = _make(test_data=None)
                return (self.list2Spectrogram(train), self.list2SS(train))
            pass 
    
    def maxNorm(self, data):
        for X in data:
            X /= X.max()
    
    def catchNone(self, x_data, y_data):
        import pandas as pd
        x_data = list(x_data)
        y_data = list(y_data)
        for num, ss in enumerate(y_data):
            if pd.isnull(ss):
                x_data.pop(num)
                y_data.pop(num)
        return np.array(x_data), np.array(y_data).astype(np.int32)
    
    def changeLabel(self, y_data):
        nr4LabelFrom = 0
        nr4LabelTo = 1
        for num, ss in enumerate(y_data):
            if ss == nr4LabelFrom:
                y_data[num] = nr4LabelTo
        return y_data-1
    
    def setDataSize(self, ss_list, target_ss, isStorchastic):
        from collections import Counter
        if isStorchastic:
            # selection based on their probabilities
            #print("確率的サンプリングを行います")
            # ss_list は record 形式で入ってくるので変換する
            ss_list = [record.ss for record in ss_list]
            loopNum = Counter(ss_list)[target_ss]
            dict_target = {target_ss : loopNum}
            ss_list = [ss for ss in ss_list if ss != target_ss]
            #print("target_ss を消した後の ss_list：", Counter(ss_list))
            # targetの睡眠段階を除いたCounter ex. Counter({2: 2945, 4: 1019, 5: 458, 3: 358, None: 8})
            ss_dict = Counter(ss_list)
            # Noneの処理
            del ss_dict[None]
            #print("Noneの処理をした後の辞書", ss_dict)
            sum_num = sum(ss_dict.values())
            ss_dict_pr = {key : ss_dict[key]/sum_num for key in ss_dict.keys()}  # value の方は辞書のままの方がkeyを用いてアクセスできる
            #print(ss_dict_pr)
            # select samples
            sampled_list = list()
            random_list = [random() for _ in range(loopNum)]
            #print(len(random_list))
            for tmp in random_list:
                bound = 0
                ss_dict_pr_keys_list = list(ss_dict_pr.keys())
                ss_dict_pr_values_list = list(ss_dict_pr.values())
                for i, k in enumerate(ss_dict_pr_values_list):
                    bound += k
                    if tmp < bound:  
                        # NOTE : NR4 がないから i+1 をしても大丈夫
                        # dict は ss_dict_pr を作る際にkeyが小さい方から昇順にソートされているのでk+1(1~5)を入れてよい
                        sampled_list.append(ss_dict_pr_keys_list[i])
                        break  #continue だと for k in range(len~)のところに戻ってしまうのでbreakじゃないとダメ
            #print("確率的サンプリング後のデータ", Counter(sampled_list))
            # サンプリングする数を表している辞書形式
            ss_dict = Counter(sampled_list)
            ss_dict.update(dict_target)
            return ss_dict
        else:
            # NOTE : ランダムサンプリングにはなってないよ（すべてのクラス同じ数作成する）
            # 400個作成する
            return {1:400, 2:400, 3:400, 4:400, 5:400}
    
    def binClassChanger(self, y_data, target_ss):
        # target_ss or non-target_ss
        from collections import Counter
        labelFrom = target_ss
        labelTo = 1
        othersLabel = 0
        for num, ss in enumerate(y_data):
            if ss == labelFrom:
                y_data[num] = labelTo
            else:
                y_data[num] = othersLabel
        print("2クラスの各ラベルの数は以下のとおりです", Counter(y_data))
        return y_data
    
    def makeSleepStageSpectrum(self):
        """全ての睡眠段階のスペクトルを返す関数．モデルの学習やテスト性能を見るために使う

        Returns:
            [type]: [description]
        """
        trainData = self.list2Spectrum(self.m_loadSleepData.trainData)
        testData = self.list2Spectrum(self.m_loadSleepData.testData)
        return trainData, testData
    
    def makeSleepStage(self):
        """全ての睡眠段階を返す関数．モデルの学習やテスト性能を見るために使う

        Returns:
            [type]: [description]
        """
        trainDataLabel = self.list2SS(self.m_loadSleepData.trainData)
        testDataLabel = self.list2SS(self.m_loadSleepData.testData)
        return trainDataLabel, testDataLabel
    
    def makeEachSleepStageTrainData(self):
        n1Data = self.list2Spectrum(self.m_loadSleepData.nr1Data)
        n2Data = self.list2Spectrum(self.m_loadSleepData.nr2Data)
        n34Data = self.list2Spectrum(self.m_loadSleepData.nr34Data)
        rData = self.list2Spectrum(self.m_loadSleepData.remData)
        wData = self.list2Spectrum(self.m_loadSleepData.wakeData)
        return n1Data, n2Data, n34Data, rData, wData
    
    def makeEachSleepStageTrainData(self):
        n1Data = self.list2Spectrum(self.m_loadSleepData.nr1DataTest)
        n2Data = self.list2Spectrum(self.m_loadSleepData.nr2DataTest)
        n34Data = self.list2Spectrum(self.m_loadSleepData.nr34DataTest)
        rData = self.list2Spectrum(self.m_loadSleepData.remDataTest)
        wData = self.list2Spectrum(self.m_loadSleepData.wakeDataTest)
        return n1Data, n2Data, n34Data, rData, wData

    def smote(self, x, y, sample_num):
        sm = SMOTE(random_state=42, k_neighbors=4)
        x, y = sm.fit_resample(x[:sample_num], y[:sample_num])
        return x, y
    
    def dataArg(self, 
                featurewise_center=True,
                featurewise_std_nomalization=True,
                #rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=False,
                vertical_flip=False):
        datagen = ImageDataGenerator(featurewise_center=featurewise_center,
                                     featurewise_std_normalization=featurewise_std_nomalization,
                                     #rotation_range=rotation_range,
                                     width_shift_range=width_shift_range,
                                     height_shift_range=height_shift_range,
                                     horizontal_flip=horizontal_flip,
                                     vertical_flip=vertical_flip)
        return datagen
    
    def make_generator(self):
        train_image_generator = ImageDataGenerator(rescale = 1./255, 
                                                   horizontal_flip=True, 
                                                   rotation_range = 45, 
                                                   zoom_range = 0.5)
        validation_image_generator = ImageDataGenerator(rescale = 1./255)
        test_image_generator = ImageDataGenerator(rescale = 1./255)
        train_data_gen = train_image_generator.flow_from_directory(batch_size=8,
                                                                   directory=self.train_dir,
                                                                   shuffle=True,
                                                                   target_size=(224, 224),
                                                                   class_mode='binary')
        val_data_gen = validation_image_generator.flow_from_directory(batch_size=8,
                                                                      directory=self.val_dir,
                                                                      target_size=(224, 224),
                                                                      class_mode='binary')
        test_data_gen = validation_image_generator.flow_from_directory(batch_size=8,
                                                                       directory = self.test_dir,
                                                                       target_size=(224, 224),
                                                                       class_mode='binary')
        return train_data_gen, val_data_gen, test_data_gen
    
    def makePreProcessedMnist(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        return (x_train, y_train), (x_test, y_test)

    def random_shift(self, x):
        """平行移動を行う前処理

        Args:
            x ([type]): [description]
        """
        import tensorflow.keras.preprocessing.image as tf_image
        import numpy
        try:
            assert type(x) == numpy.ndarray
        except:
            print("xはnumpyの形式で渡してください")
        shifted_x = tf_image.random_shift(x, wrg=0.3, hrg=0.3)
        return shifted_x
        
    def plot_images(self, images_arr):
        fig, axes = plt.subplots(1, 5, figsize=(20,20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()
        
    def makeConfusionMatrix(self, x, y, model, using_pandas = False):
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
    
    def saveImage2Wandb(self, image, dir2 = "confusion_matrix", fileName = "cm", to_wandb = False):
        sns.heatmap(image, annot = True, cmap = "Blues", fmt = "d")
        path = self.figure_dir + os.path.join(dir2, fileName+".png")
        plt.savefig(path)
        if to_wandb:
            im_read = plt.imread(path)
            wandb.log({f"{dir2}":[wandb.Image(im_read, caption = f"{fileName}")]})
        plt.clf()
    
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
        
    def makeNetworkGraph(self, model, dir2 = "test", fileName = "test"):
        from keras.utils import plot_model
        from IPython.display import SVG
        from keras.utils.vis_utils import model_to_dot
        path = self.figure_dir + os.path.join(dir2, fileName+".png")
        SVG(data = model_to_dot(model).create(prog = "dot", format = "svg"))
    
    def checkPath(self, path):
        if not os.path.exists(path):
            print(f"フォルダが見つからなかったので以下の場所に作成します．\n {path}")
            os.mkdir(path)

# ================================================ #
# *          様々な自分用の便利関数
# ================================================ #

class Utils():
    def __init__(self) -> None:
        self.id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.user_dir = os.environ['userprofile']
        self.defaultPath = os.path.join(self.user_dir,
                                        "sleep_study",
                                        "figures",
                                        "tmp")
        self.o_fileReader = FileReader()
        self.date = self.o_fileReader.date
        self.name_list = self.o_fileReader.name_list
        self.name_dict = self.o_fileReader.name_dict
        pass
    
    def dumpWithPickle(self, data, finds_dir_obj, file_name):
        
        filePath = os.path.join(finds_dir_obj.returnDataPath(),
                                "pre_processed_data",
                                file_name+'_'+self.id+'.sav')
        pickle.dump(data, open(filePath, 'wb'))
        
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
    
    def show_properties(self, df_list):
        # df_list はlist型の9x5の構造になっているようにする
        assert type(df_list) == list
        assert df_list.__len__() == 9
        assert df_list[0].__len__() == 5
        from subjectsList import SubjectsList
        returned_dict = dict()
        name_list = SubjectsList().nameList
        ss_list = SubjectsList().ssList
        for _df_list, name in zip(df_list, name_list):
            for df, ss in zip(_df_list, ss_list):
                dict_key = name+"_"+ss
                assert df.shape == (2,2)
                returned_dict.update({dict_key:self.compute_properties(df)})
        return returned_dict

# ================================================ #
# *         不均衡データに対する前処理
# ================================================ #

class ImbalancedData():
    def __init__(self) -> None:
        pass

# ================================================ #
# *            試験用メイン関数
# ================================================ #

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
    from my_model import *
    import tensorflow as tf
    # physical_devices = tf.config.list_physical_devices("GPU")
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    my_setting.SetsPath().set()
    o_findsDir = my_setting.FindsDir("sleep")
    o_preProcess = PreProcess(project="sleep_study", input_file_name="H_Li")
    (x_train, y_train) = o_preProcess.loadData(is_split=True)
    model.summary()
    hidden = model(x_train)
    hidden.shape
    tmp = x_train.copy()
    o_preProcess.maxNorm(x_train)
    o_utils = Utils()
    o_utils.showSpectrogram(x_train, tmp)
    
    # 新しいデータ拡張のテスト
    
    
    
