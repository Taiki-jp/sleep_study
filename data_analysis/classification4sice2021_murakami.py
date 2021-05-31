# ================================================ #
# *             Import Libraries
# ================================================ #

from pandas.core.algorithms import mode
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
from pprint import pprint

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

def main(best_f_epochs, is_attention=True, name="H_Murakami",
         is_load_attention_by_name=False, loop_start_of_attention=0, 
         is_wandb=True, tags=list()):
    
    # データセットの作成
    o_findsDir = FindsDir("sleep")
    o_preProcess = PreProcess(project=o_findsDir.returnDirName(), input_file_name=name)
    (x_test, y_test) = o_preProcess.loadData(is_split=True)
    (x_test, y_test) = o_preProcess.catchNone(x_test, y_test)
    o_preProcess.maxNorm(x_test)
    assert type(best_f_epochs) is list or tuple or dict and len(best_f_epochs) == 5
    
    # 読み込みのパスの設定
    if is_attention:
        modelDirPath = os.path.join(os.environ["sleep"], "models", f"{name}", "attention", "*")
        tags.append("attention")
    else:
        modelDirPath = os.path.join(os.environ["sleep"], "models", f"{name}", "no-attention", "*")
        tags.append("no-attention")
    modelList = glob(modelDirPath)
    print("*** this is model list ***")
    pprint(modelList)  # ss_1 ~ ss_5 まで入っている
    target_ss_list = ["nr34", "nr2", "nr1", "rem", "wake"]
    ss_num_list = [1, 2, 3, 4, 5]
    
    # モデルのパスの設定
    # best_f_epochs = (1, 8, 1, 7, 10)
    nr34_path = os.path.join(modelList[0], f"cp-{best_f_epochs[0]:04d}.ckpt")
    nr2_path = os.path.join(modelList[1], f"cp-{best_f_epochs[1]:04d}.ckpt")
    nr1_path = os.path.join(modelList[2], f"cp-{best_f_epochs[2]:04d}.ckpt")
    rem_path = os.path.join(modelList[3], f"cp-{best_f_epochs[3]:04d}.ckpt")
    wake_path = os.path.join(modelList[4], f"cp-{best_f_epochs[4]:04d}.ckpt")

    # モデルの読み込み
    model_nr34 = tf.keras.models.load_model(nr34_path)
    model_nr2 = tf.keras.models.load_model(nr2_path)
    model_nr1 = tf.keras.models.load_model(nr1_path)
    model_rem = tf.keras.models.load_model(rem_path)
    model_wake = tf.keras.models.load_model(wake_path)

    datasize = y_test.shape[0]
    ss_num = 5
    df_label = pd.DataFrame(np.zeros(datasize*ss_num).reshape(datasize, ss_num))
    labels_list = ["wake", "rem", "nr1", "nr2", "nr34"]

    # モデルのリストを作成してループを回す NOTE : wakeの方から入っていることに注意
    model_list = [model_wake, model_rem, model_nr1, model_nr2, model_nr34]
    for num, model in enumerate(model_list):
        pr = tf.math.softmax(model.predict(x_test))
        for k in range(datasize):
            df_label[num][k] = 1 if pr[k, 1].numpy() > pr[k, 0].numpy() else 0
    
    # TODO : 正解ラベルを追加する
    df_true = pd.DataFrame(y_test)
    df_true_path = os.path.join(os.environ["sleep"], "datas", "y_true.csv")
    tmp_path = os.path.join(os.environ["sleep"], "datas", "all_sleep_stages.csv")
    df_true.to_csv(df_true_path, mode='a')
    df_label.to_csv(tmp_path, mode='a')

    # NNの自信のある部分の一致率
    def _get_how_many_rows_selected_in_each_condition(verbose = 0):
        """[summary]

        Returns:
            [type]: 
            [一つだけがtrueの時の行番号：catched_rows, 
            1個もtrueがない時の行番号：no_labels_selected_rows，
            複数のtrueが存在したときの行番号：multiple_labels_selected_rows]
        """
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
        if verbose == 0:
            print("自信をもって判断された数", len(catched_rows), 
                  "一つもtrueがない時の数", len(no_labels_selected_rows), 
                  "複数のラベルが選ばれた数", len(multiple_labels_selected_rows))
        return (catched_rows, no_labels_selected_rows, multiple_labels_selected_rows)
    
    (one_labels, no_labels, mul_labels) = _get_how_many_rows_selected_in_each_condition()

    def _select_one_label_from_multiples(multi_list, df):
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
                    pr_array[4] = tf.math.softmax(model_wake.predict(tf.expand_dims(x_test[data_num], axis=0)))[0][1].numpy()
                elif label == 1:  #rem
                    pr_array[3] = tf.math.softmax(model_rem.predict(tf.expand_dims(x_test[data_num], axis=0)))[0][1].numpy()
                elif label == 2:  #nr1
                    pr_array[2] = tf.math.softmax(model_nr1.predict(tf.expand_dims(x_test[data_num], axis=0)))[0][1].numpy()
                elif label == 3:  #nr2
                    pr_array[1] = tf.math.softmax(model_nr2.predict(tf.expand_dims(x_test[data_num], axis=0)))[0][1].numpy()
                elif label == 4:  #nr34
                    pr_array[0] = tf.math.softmax(model_nr34.predict(tf.expand_dims(x_test[data_num], axis=0)))[0][1].numpy()
                else:
                    print("error handle")
                    sys.exit(1)
            max_label = np.argmax(pr_array)   # 最大の要素のインデックスを返す
            tmp.append(max_label+1)   # max_labelは要素数なので睡眠段階に直すために+1をする
        return tmp
    
    selected_labels_in_multiples = _select_one_label_from_multiples(multi_list=mul_labels, df=df_label)

    def _count_mul():
        one_true_patterns_correct = 0
        for i, row in enumerate(mul_labels):
            if selected_labels_in_multiples[i] == y_test[row]:
                one_true_patterns_correct += 1
        one_true_patterns_num = len(mul_labels)
        return (one_true_patterns_correct, one_true_patterns_num)
    
    true_num_mul, all_num_mul = _count_mul()
    
    def _count_one():
        one_true_patterns_num = len(one_labels)
        one_true_patterns_correct = 0
        nd_label = np.array(df_label)
        for row in one_labels:
            if nd_label[row][y_test[row]*(-1)]:
                one_true_patterns_correct += 1
        return (one_true_patterns_correct, one_true_patterns_num)
    
    true_num_one, all_num_one = _count_one()
    
    # それぞれの睡眠段階がどれだけtrueと言われたかを示す指標になっている
    wake_true = df_label[0].sum()
    rem_true = df_label[1].sum()
    nr1_true = df_label[2].sum()
    nr2_true = df_label[3].sum()
    nr34_true = df_label[4].sum()
    
    if is_wandb:
        tags.append("testそのまま")
        wandb.init(name=name, project="sleep",
                   config={"name":name,
                           "data-size":datasize,
                           "acc_selected_once":true_num_one/all_num_one,
                           "acc_selected_mul":true_num_mul/all_num_mul,
                            "one_datas_selected":all_num_one,
                            "mul_datas_selected":all_num_mul,
                            "only_wake_selected":wake_true,
                            "only_rem_selected":rem_true,
                            "only_nr1_selected":nr1_true,
                            "only_nr2_selected":nr2_true,
                            "only_nr34_selected":nr34_true,
                            "no_labels_selected":len(no_labels)},
                   tags=tags)
        wandb.finish()


# ================================================ #
# *                    実行部
# ================================================ #

if __name__ == "__main__":
    isAttention = False
    if isAttention:
        best_f_epochs = (1, 8, 1, 7, 13)
    else:
        best_f_epochs = (1, 4, 6, 1, 13)
    main(best_f_epochs=best_f_epochs, is_attention=isAttention)
    