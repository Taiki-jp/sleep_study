# ================================================ #
# *         Import Some Libraries
# ================================================ #
#%%
from my_setting import FindsDir, SetsPath
SetsPath().set()
import os, sys
import tensorflow as tf
from utils import PreProcess, Utils
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["WANDB_SILENT"] = "true"

try:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("GPUが設定されていません")
    
tf.keras.backend.set_floatx('float32')


# ================================================ #
# *          モデルの読み込みと作成
# ================================================ #
#%%
o_FindsDir = FindsDir("sleep")
modelDirPath = o_FindsDir.modelsDir
modelList = glob(modelDirPath+'/*')
"""
tmp = list()
for model in modelList[45::]:
    if model != '\\\\gamma\\Workspace\\TaikiSenju\\sleep_study\\models\\20210215-150631':
        tmp.append(model)
    else:
        tmp.append(model)
        break
modelList = tmp
"""
modelList = modelList[45:]
# assert modelList[-1] == os.path.join(modelDirPath, "20210215-150631")
# モデルリストの頭の方にLiさんのデータ後ろの方にYamamotoさんのデータが入っている
# ================================================ #
# *          競合したときの処理方法
# ================================================ #
#%%
#inputFileName = input("*** 被験者データを入れてください *** \n")
for loop_num, name in enumerate(Utils().name_list[::-1]):
    #name = "H_Li"
    #wandb.init(project='sleep', name = f"{name}")
    #print("だれだれの実験をやっています", name)
    m_preProcess = PreProcess(project=o_FindsDir.returnDirName(), input_file_name=name)
    (x_test, y_test) = m_preProcess.loadData(is_split=True)
    # x_test に関しては前処理が必要(Noneの処理，x_testの正規化)
    (x_test, y_test) = m_preProcess.catchNone(x_test, y_test)
    m_preProcess.maxNorm(x_test)
    # データサイズを入れてこれを基にデータフレームを（行数, 睡眠段階数）作る
    datasize = y_test.shape[0]
    ss_num = 5
    df_label = pd.DataFrame(np.zeros(datasize*ss_num).reshape(datasize, ss_num))

    labels_list = ["wake", "rem", "nr1", "nr2", "nr34"]
    list4read_models = np.arange(-1, -6, -1) - 5*loop_num
    for num, i in tqdm(enumerate(list4read_models)):
        print("読み込むモデル名:", modelList[i])
        model = tf.keras.models.load_model(modelList[i])
        pr = tf.math.softmax(model.predict(x_test))
        for k in range(datasize):
            try:
                df_label[num][k] = True if pr[k, 1].numpy() > pr[k, 0].numpy() else False
            except:
                print(k, num, i)
                sys.exit(1)
    # 複数のモデルを作成する（複数のラベルから最大のコンフィデンスのものを選択するため）
    model4wake = tf.keras.models.load_model(modelList[list4read_models[0]])
    model4rem = tf.keras.models.load_model(modelList[list4read_models[1]])
    model4nr1 = tf.keras.models.load_model(modelList[list4read_models[2]])
    model4nr2 = tf.keras.models.load_model(modelList[list4read_models[3]])
    model4nr34 = tf.keras.models.load_model(modelList[list4read_models[4]])
    # NNの自信のある部分の一致率
    nd_label = np.array(df_label)
    catched_rows = list()   # 1個だけがtrueの時は行番号をcatched_rowsに入れる
    no_labels_selected_rows = list()   # 1個もtrueがない時は行番号をno_labels_selected_rowsに入れる
    multiple_labels_selected_rows = list()   # 複数のtrueが存在したときは行番号を入れる
    for row, _ in enumerate(range(datasize)):
        if nd_label[row].sum() == 1:
            catched_rows.append(row)
        elif nd_label[row].sum() == 0:
            no_labels_selected_rows.append(row)
        else:
            multiple_labels_selected_rows.append(row)
    def select_one_label_from_multiples(multi_list, df):
        """睡眠段階を返す

        Args:
            multi_list ([type]): [複数ラベルが選択された時のx_testの行番号]
            df ([type]): [(datasize, 睡眠段階数)のデータフレームで2クラス分類のモデルがtrueと選択したものがtrueになっている]

        Returns:
            [type]: [description]
        """
        df2nd = np.array(df)
        # 複数のラベルが選ばれているとき選ばれた(true)のラベルに対して確率を計算して最大のものを選択する
        tmp = list()   # 複数ラベルから一つを選ぶ．どの行（データ）(multi*)，土の睡眠段階がtrueか(df2nd)が必要
        for data_num in multi_list:
            # ラベルがtrueのところの要素を返す(0:w,..., 4:nr34)
            check_labels = [i for i in range(5) if df2nd[data_num, i] == True] #ok
            pr_array = np.zeros(shape=(5))
            for label in check_labels:   # trueの列番号0 - 4 に対して処理を行う
                if label == 0:  #wake
                    pr_array[4] = tf.math.softmax(model4wake.predict(tf.expand_dims(x_test[data_num], axis=0)))[0][1].numpy()
                elif label == 1:  #rem
                    pr_array[3] = tf.math.softmax(model4rem.predict(tf.expand_dims(x_test[data_num], axis=0)))[0][1].numpy()
                elif label == 2:  #nr1
                    pr_array[2] = tf.math.softmax(model4nr1.predict(tf.expand_dims(x_test[data_num], axis=0)))[0][1].numpy()
                elif label == 3:  #nr2
                    pr_array[1] = tf.math.softmax(model4nr2.predict(tf.expand_dims(x_test[data_num], axis=0)))[0][1].numpy()
                elif label == 4:  #nr34
                    pr_array[0] = tf.math.softmax(model4nr34.predict(tf.expand_dims(x_test[data_num], axis=0)))[0][1].numpy()
                else:
                    print("error handle")
                    sys.exit(1)
            max_label = np.argmax(pr_array)   # 最大の要素のインデックスを返す
            tmp.append(max_label+1)   # max_labelは要素数なので睡眠段階に直すために+1をする
        return tmp
    selected_labels_in_multiples = select_one_label_from_multiples(multi_list=multiple_labels_selected_rows, df=df_label)
    
    # 複数ラベルの正解ラベルとの比較
    # TODO : 確率が0になる
    one_true_patterns_correct = 0
    for i, row in enumerate(multiple_labels_selected_rows):
        
        if selected_labels_in_multiples[i] == y_test[row]:
            one_true_patterns_correct += 1
    one_true_patterns_num = len(multiple_labels_selected_rows)
    #print("NNが複数をもって判断したときの5段階一致率は", one_true_patterns_correct/one_true_patterns_num)
    wandb.config.update({"how_many_datas_selected_when_mul":one_true_patterns_num})
    wandb.config.update({"acc_selected_mul":one_true_patterns_correct/one_true_patterns_num})

    # 一つしか２クラス分類がtrueといわなかったときの正解ラベルとの比較
    # 一つしかtrueといわなかったものの数
    one_true_patterns_num = len(catched_rows)
    one_true_patterns_correct = 0
    for row in catched_rows:
        #print("true", y_test[row])
        #print("predicted", nd_label[row])
        if nd_label[row][y_test[row]*(-1)]:
            one_true_patterns_correct += 1
    # 確率を表示
    #print("NNが自信をもって判断したときの5段階一致率は", one_true_patterns_correct/one_true_patterns_num)
    # ちなみに何個そういうときがあるかを見る
    #print("全体で何通りそのようなデータがあったか", one_true_patterns_num)
    #trueの数の確認
    wake_true = df_label[0].sum()
    rem_true = df_label[1].sum()
    nr1_true = df_label[2].sum()
    nr2_true = df_label[3].sum()
    nr34_true = df_label[4].sum()
    #print(wake_true, rem_true, nr1_true, nr2_true, nr34_true)   
    wandb.config.update(
        {
            "name":name,
            "acc_selected_once":one_true_patterns_correct/one_true_patterns_num,
            "how_many_datas_selected":one_true_patterns_num,
            "only_wake_selected":wake_true,
            "only_rem_selected":rem_true,
            "only_nr1_selected":nr1_true,
            "only_nr2_selected":nr2_true,
            "only_nr34_selected":nr34_true
        }
    )
    wandb.finish()
   