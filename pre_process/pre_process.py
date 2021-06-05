import datetime, sys, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # tensorflow を読み込む前のタイミングですると効果あり
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
from random import shuffle, choices, random, seed
import tensorflow as tf
import tensorflow.keras.preprocessing.image as tf_image
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class PreProcess():
    
    def __init__(self, load_sleep_data=None):
        seed(0)
        self.date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.load_sleep_data = load_sleep_data
        if self.load_sleep_data is not None:
            self.fr = self.load_sleep_data.fr
            self.sl = self.load_sleep_data.sl
            self.data_type = self.load_sleep_data.data_type
            self.name_list = self.load_sleep_data.sl.name_list
            self.name_dict = self.load_sleep_data.sl.name_dict
            self.ss_list = self.load_sleep_data.sl.ss_list
            self.verbose = self.load_sleep_data.verbose
        else:
            self.fr = None
            self.sl = None
            self.data_type = None
            self.name_list = None
            self.name_dict = None
            self.ss_list = None
            self.verbose = 0
    
    # データセットの作成（この中で出来るだけ正規化なども含めて終わらせる）
    def make_dataset(self, train=None, test=None,
                     is_set_data_size = True, 
                     target_ss = None, 
                     is_storchastic = True, 
                     is_shuffle = True,
                     is_multiply = False,
                     mul_num = None,
                     each_data_size=1000,
                     class_size=5,
                     normalize=True,
                     catch_none=True,
                     insert_channel_axis=True,
                     to_one_hot_vector=True,
                     pse_data = False):

        if pse_data:
            print("仮の睡眠データを作成します")
            return self.make_pse_sleep_data(n_class=class_size)
    
        if is_set_data_size:
            
            if self.verbose == 0:
                print("訓練データのサイズを揃えます")
            
            # 各睡眠段階のサイズを決定する
            ss_dict_train = self.set_datasize(train, target_ss=target_ss, isStorchastic=is_storchastic, 
                                              each_data_size=each_data_size, class_size=class_size)
            ss_dict_test = Counter([record.ss for record in test])
            
            if self.verbose == 0:
                print("訓練データの各睡眠段階（補正前）", Counter([record.ss for record in train]))
                print("訓練データの各睡眠段階（補正後）", ss_dict_train)
            
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
            
            train = _storchastic_sampling(data=ss_dict_train, target_records=train, is_test=False)
            test = _storchastic_sampling(data=ss_dict_test, target_records=test, is_test=True)
            
            if is_shuffle:
                print("- 訓練データをシャッフルします")
                shuffle(train)
                
        else:
            if self.verbose == 0:
                print('データサイズをそろえずにデータセットを作成します')
        
        # TODO : スペクトログラムかスペクトラム化によって呼び出す関数を場合分け
        y_train = self.list2SS(train)
        y_test = self.list2SS(test)
        if self.data_type == "spectrum":
            x_train = self.list2Spectrum(train)
            x_test = self.list2Spectrum(test)
        elif self.data_type == "spectrogram":
            x_train = self.list2Spectrogram(train)
            x_test = self.list2Spectrogram(test)
        else:
            print("spectrum or spectrogramを指定してください")
            sys.exit(1)
        
        # max正規化をするかどうか
        if normalize==True:
            print("- max正規化を行います")
            self.max_norm(x_train)
            self.max_norm(x_test)

        # Noneの処理をするかどうか
        if catch_none==True:
            print("- noneの処理を行います")
            x_train, y_train = self.catch_none(x_train, y_train)
            x_test, y_test = self.catch_none(x_test, y_test)
        
        # 睡眠段階のラベルを0 -（クラス数-1）にする
        y_train = self.change_label(y_data=y_train, n_class=class_size, target_class=target_ss)
        y_test = self.change_label(y_data=y_test, n_class=class_size, target_class=target_ss)
        
        # inset_channel_axis
        if insert_channel_axis:
            print("- チャンネル方向に軸を追加します")
            x_train = x_train[:,:,:,np.newaxis]  #.astype('float32')
            x_test = x_test[:,:,:,np.newaxis] #.astype('float32')

        if self.verbose == 0:
            print("*** 全ての前処理後（one-hotを除く）の訓練データセット（確認用） *** \n", Counter(y_train))

        # convert to one-hot vector
        if to_one_hot_vector:
            print("- one-hotベクトルを出力します")
            y_train = tf.one_hot(y_train, class_size)
            y_test = tf.one_hot(y_test, class_size)
            
        return (x_train, y_train), (x_test, y_test)

    # recordからスペクトラムの作成
    def list2Spectrum(self, list_data):
        return np.array([data.spectrum for data in list_data])
    
    # recordから睡眠段階の作成
    def list2SS(self, list_data):
        return np.array([k.ss for k in list_data])
    
    # recordからスペクトログラムの作成
    def list2Spectrogram(self, list_data):
        return np.array([data.spectrogram for data in list_data])
    
    # 訓練データとテストデータをスプリット（ホールドアウト検証）
    def split_train_test_data(self, name=None, load_all=False,test_name=None, pse_data=False):
        
        if pse_data:
            print("仮データです．データの読み込みは行いません")
            return None, None
        # 一人の被験者を入れたときはこの人だけ読み込む
        if name is not None:
            datas = self.load_sleep_data.load_data(name=name)
            return datas
        # 全員の被験者データを読み込む
        if load_all:
            datas = self.load_sleep_data.load_data(load_all=load_all)
        else:
            print("name, load_allのどちらかを指定してください")
            sys.exit(1)
        
        # NOTE : 一人以上の被験者を読み込んでいるときのみ以下実行可能
        
        # 被験者名を指定して訓練データとテストデータを分ける
        records_train = list()
        records_test = list()
        
        # NOTE : dataの順番と被験者の順番が等しいからできる処理
        for id, data in enumerate(datas):
            if test_name == self.name_list[id]:
                records_test.extend(data)
                print(f"testデータは{id}番目の{test_name}です")
            else:
                records_train.extend(data)
        
        return records_train, records_test
    
    # 訓練データとテストデータをスプリット（ホールドアウト検証（旧バージョン））
    def split_train_test_from_records(self, records, test_id, pse_data):
        if pse_data:
            print("仮データのためスキップします")
            return (None, None)
        test_records = records[test_id]
        train_records = list()
        for i in range(9):
            train_records.extend(records[i])
        return (train_records, test_records)
    
    # 訓練データのサイズをセットする
    def set_datasize(self, ss_list, target_ss, isStorchastic, each_data_size, class_size):

        if isStorchastic:
            print("確率的サンプリングを実装し直しましょう")
            sys.exit(1)
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
            if class_size == 5:  # NR34, NR2, NR1, REM, WAKE
                return {1:each_data_size, 2:each_data_size, 3:each_data_size, 4:each_data_size, 5:each_data_size}
            elif class_size == 4:  # NR34, NR12, REM, WAKE
                return {1:each_data_size, 2:each_data_size, 3:each_data_size, 4:each_data_size}
            elif class_size == 3:  # NREM, REM, WAKE
                return {1:each_data_size, 2:each_data_size, 3:each_data_size}
            elif class_size == 2:  # TARGET, NON-TARGET
                return {1:each_data_size, 2:each_data_size}
            else:
                print("クラスサイズは5以下2以上に入れてください")
                sys.exit(1)    
    
    # 各スペクトルの最大値で正規化
    def max_norm(self, data):  
        for X in data:  # TODO : 全体の値で割るようなこともする
            X /= X.max()
    
    # NONEの睡眠段階をキャッチして入力データごと消去
    def catch_none(self, x_data, y_data):
        import pandas as pd
        x_data = list(x_data)
        y_data = list(y_data)
        for num, ss in enumerate(y_data):
            if pd.isnull(ss):
                x_data.pop(num)
                y_data.pop(num)
        return np.array(x_data), np.array(y_data).astype(np.int32)
    
    # ラベルをクラス数に合わせて変更
    def change_label(self, y_data, n_class, target_class=None):
        nr4_label_from = 0
        nr3_label_from = 1
        nr2_label_from = 2
        nr1_label_from = 3
        rem_label_from = 4
        wake_label_from = 5
        # 5クラス分類の時（0:nr34, 1:nr2, 2:nr1, 3:rem, 4:wake）
        if n_class == 5:
            nr4_label_to = 0
            nr3_label_to = 0
            nr2_label_to = 1
            nr1_label_to = 2
            rem_label_to = 3
            wake_label_to = 4   
        # 4クラス分類の時（0:nr34, 1:nr12, 2:rem, 3:wake）
        elif n_class == 4:
            nr4_label_to = 0
            nr3_label_to = 0
            nr2_label_to = 1
            nr1_label_to = 1
            rem_label_to = 2
            wake_label_to = 3
        # 3クラス分類の時（0:nrem, 1:rem, 2:wake）
        elif n_class == 3:
            nr4_label_to = 0
            nr3_label_to = 0
            nr2_label_to = 0
            nr1_label_to = 0
            rem_label_to = 1
            wake_label_to = 2
        # 2クラス分類とき（0:non_target, 1:target）
        elif n_class == 2:
            assert target_class is not None
            nr4_label_to = 0 if target_class != "nr4" else 1
            nr3_label_to = 0 if target_class != "nr3" else 1
            nr2_label_to = 0 if target_class != "nr2" else 1
            nr1_label_to = 0 if target_class != "nr1" else 1
            rem_label_to = 0 if target_class != "rem" else 1
            wake_label_to = 0 if target_class != "wake" else 1
        else:
            print("正しいクラス数を指定してください")
            sys.exit(1)  
        
        # ラベル変更
        for num, ss in enumerate(y_data):
            if ss == nr4_label_from:
                y_data[num] = nr4_label_to
            elif ss == nr3_label_from:
                y_data[num] = nr3_label_to
            elif ss == nr2_label_from:
                y_data[num] = nr2_label_to
            elif ss == nr1_label_from:
                y_data[num] = nr1_label_to
            elif ss == rem_label_from:
                y_data[num] = rem_label_to
            elif ss == wake_label_from:
                y_data[num] = wake_label_to
        return y_data

    # データ拡張の一つ
    def smote(self, x, y, sample_num):
        sm = SMOTE(random_state=42, k_neighbors=4)
        x, y = sm.fit_resample(x[:sample_num], y[:sample_num])
        return x, y
    
    # tutorialにありそうなデータ拡張
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
    
    # 仮データの作成
    def make_pse_sleep_data(self, n_class):
        batch_size = 1000
        if self.data_type == "spectrum":
            print("spectrum型の仮データを作成します")
            input_shape = (batch_size, 512, 1)
        elif self.data_type == "spectrogram":
            print("spectrogram型の仮データを作成します")
            input_shape  = (batch_size, 128, 512, 1)
        # x_train = tf.random.normal(shape=input_shape)
        x_train = tf.random.uniform(shape=input_shape,minval=-1,maxval=1)
        y_train = np.random.randint(0, n_class-1, size=batch_size)
        # x_test = tf.random.normal(shape=input_shape)
        x_test = tf.random.uniform(shape=input_shape,minval=-1,maxval=1)
        y_test = np.random.randint(0, n_class-1, size=batch_size)
        return(x_train, y_train), (x_test, y_test)
     
    # 平行移動を行う前処理
    def random_shift(self, x, wrg, hrg):
        try:
            assert type(x) == np.ndarray
        except:
            print("xはnumpyの形式で渡してください")
        shifted_x = tf_image.random_shift(x, wrg=wrg, hrg=hrg)
        return shifted_x

if __name__ == "__main__":
    from pre_process.load_sleep_data import LoadSleepData
    load_sleep_data = LoadSleepData(data_type="spectrogram", verbose=0)
    pre_process = PreProcess(load_sleep_data)
    # 全員読み込む
    #records = load_sleep_data.load_data(load_all=True)
    # テストと訓練にスプリット
    #(train, test) = pre_process.split_train_test_from_records(records=records, test_id=0)
    # 読み込みとsplitを同時に行う
    (train, test) = pre_process.split_train_test_data(load_all=True, test_name="H_Li")
    (x_train, y_train), (x_test, y_test) = pre_process.make_dataset(train=train, test=test, is_storchastic=False)
    print(x_train.shape)
