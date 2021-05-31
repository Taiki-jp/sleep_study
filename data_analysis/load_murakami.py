# ================================================ #
# *         Import Some Libraries
# ================================================ #

from my_setting import FindsDir, SetsPath
SetsPath().set()
import os, sys
import tensorflow as tf
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.keras.backend.set_floatx('float32')
from utils import PreProcess, Utils
from glob import glob
from pprint import pprint
import numpy as np

# ================================================ #
# *          モデルの読み込みと作成
# ================================================ #

def main(best_f_epochs, is_attention = True, attention_name = "my_attention2d", name = "H_Murakami", is_load_attention_by_name = False, loop_start_of_attention=0):
    o_findsDir = FindsDir("sleep")
    o_preProcess = PreProcess(project=o_findsDir.returnDirName(), input_file_name=Utils().name_dict)
    (train, test) = o_preProcess.loadData(is_split=True)
    try:
        assert o_preProcess.test_data_for_wandb == name
    except:
        print("main関数で指定されたテストデータとloadDataで指定されたデータが異なります")
        sys.exit(1)
    assert type(best_f_epochs) is list or tuple or dict and len(best_f_epochs) == 5
    
    # 読み込みのパスの設定
    if is_attention:
        modelDirPath = os.path.join(os.environ["sleep"], "models", f"{name}", "attention", "*")
    else:
        modelDirPath = os.path.join(os.environ["sleep"], "models", f"{name}", "no-attention", "*")
    modelList = glob(modelDirPath)
    print("*** this is model list ***")
    pprint(modelList)  # ss_1 ~ ss_5 まで入っている
    target_ss_list = ["nr34", "nr2", "nr1", "rem", "wake"]
    ss_num_list = [1, 2, 3, 4, 5]
    
    for (target_ss, ss_dir_path, best_epoch, sleep_stage) in zip(ss_num_list[loop_start_of_attention::], modelList[loop_start_of_attention::], best_f_epochs[loop_start_of_attention::], target_ss_list[loop_start_of_attention::]):
        
        load_path = os.path.join(ss_dir_path, f"cp-{best_epoch:04d}.ckpt")
        print(f"{load_path}を読み込んでいます")
        model = tf.keras.models.load_model(load_path)
        # NOTE : 読み込みが層名が良いか，インデックスで指定した方が良いかは分からない
        if is_load_attention_by_name:
            new_model = tf.keras.Model(model.input, model.get_layers(attention_name))
        else:
            new_model = tf.keras.Model(model.input, model.layers[3].output)
        new_model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer = tf.keras.optimizers.Adam(), metrics = ["accuracy"])
        (x_train, y_train), (x_test, y_test) = o_preProcess.makeDataSet(train, test, is_split=True, target_ss=target_ss)
        o_preProcess.maxNorm(x_train)
        o_preProcess.maxNorm(x_test)
        (x_train, y_train) = o_preProcess.catchNone(x_train, y_train)
        (x_test, y_test) = o_preProcess.catchNone(x_test, y_test)
        y_train = o_preProcess.binClassChanger(y_train, target_ss)
        y_test = o_preProcess.binClassChanger(y_test, target_ss)

        # 関数内関数定義
        def _attention(x_data, y_data, mode):
            """アテンションを描画する

            Args:
                data ([tuple]): [train or test]
            """
            non_target = list()
            target = list()

            for num, ss in enumerate(y_data):
                if ss == 0:
                    non_target.append(x_data[num])
                elif ss == 1:
                    target.append(x_data[num])

            non_target = np.array(non_target)
            target = np.array(target)

            attentionArray = []
            confArray = []

            convertedArray = [non_target, target]

            for num, inputs in enumerate(convertedArray):
                attention = new_model.predict(inputs)
                if num == 0:
                    labelNum = 0
                elif num == 1:
                    labelNum = 1
                else:
                    labelNum = None
                conf = tf.math.softmax(model.predict(inputs))[:, labelNum]
                attentionArray.append(attention)
                confArray.append(conf)

            pathRoot = os.path.join(os.environ["sleep"],"figures/", f"{name}")
            o_preProcess.checkPath(pathRoot)
            pathRoot = os.path.join(pathRoot, f"{mode}_attention_{sleep_stage}/")
            o_preProcess.checkPath(pathRoot)
            savedDirList = ["non_target", "target"]
            savedDirList = [pathRoot + savedDir for savedDir in savedDirList]
            for num, target in enumerate(attentionArray):
                o_preProcess.checkPath(savedDirList[num])
                o_preProcess.simpleImage(image_array = target,
                                         row_image_array = convertedArray[num],
                                         file_path = savedDirList[num],
                                         x_label = "time[s]",
                                         y_label = "frequency[Hz]",
                                         title_array = confArray[num])
        
        # 関数内関数の利用
        _attention(x_test, y_test, mode="test")
        # 時間に余裕があればこっちをコメントアウトして実行して
        _attention(x_train, y_train, mode="train")

# ================================================ #
# *              実行部
# ================================================ #

if __name__ == '__main__':
    main(best_f_epochs=(1, 8, 1, 7, 10), loop_start_of_attention=0)