import datetime
import os
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from data_analysis.py_color import PyColor
from data_analysis.utils import Utils
from nn.utils import load_model
from pre_process.pre_process import PreProcess
from pre_process.utils import set_seed


# TODO: 使っていない引数の削除
def main(
    train: list,
    test: list,
    pre_process: PreProcess,
    n_class: int = 5,
    pse_data: bool = False,
    test_name: str = None,
    date_id: dict = None,
    sample_size: int = 0,
    batch_size: int = 0,
    utils: Utils = None,
    is_under_4hz: bool = False,
):

    # データセットの作成
    ((_, _), (_, _), (x_test, y_test),) = pre_process.make_dataset(
        train=train,
        test=test,
        is_storchastic=False,
        pse_data=pse_data,
        to_one_hot_vector=False,
        each_data_size=sample_size,
        is_under_4hz=is_under_4hz,
    )

    model = load_model(
        loaded_name=test_name, model_id=date_id, n_class=n_class, verbose=1
    )

    # NOTE : そのためone-hotの状態でデータを読み込む必要がある
    # TODO: このコピーいる？
    evidence = model.predict(x=x_test, batch_size=batch_size)
    _, _, _, y_pred = utils.calc_enn_output_from_evidence(evidence)
    # 睡眠段階の比較
    utils.compare_ss(y_true=y_test[0], y_pred=y_pred, test_name=test_name)
    # 混合行列の作成
    cm = utils.make_confusion_matrix(
        y_true=y_test[0], y_pred=y_pred, n_class=5
    )
    __cm = confusion_matrix(y_true=y_test[0], y_pred=y_pred)
    utils.save_image2Wandb(
        image=cm,
        to_wandb=False,
        is_specific_path=True,
        specific_name=test_name,
    )
    confdiag = np.eye(len(__cm)) * __cm
    np.fill_diagonal(__cm, 0)

    ss_dict = Counter(y_test[0])

    eps4zero_div = 0.001
    if __cm.shape[0] == 5:
        rec_log_dict = {
            "rec_" + ss_label: confdiag[i][i] / (ss_dict[i])
            for (ss_label, i) in zip(
                ["nr34", "nr2", "nr1", "rem", "wake"], range(5)
            )
        }
        pre_log_dict = {
            "pre_"
            + ss_label: confdiag[i][i] / (sum(__cm[:, i]) + confdiag[i][i])
            for (ss_label, i) in zip(
                ["nr34", "nr2", "nr1", "rem", "wake"], range(5)
            )
        }
        f_m_log_dict = {
            "f_m_" + ss_label: rec * pre * 2 / (rec + pre)
            for (rec, pre, ss_label) in zip(
                rec_log_dict.values(),
                pre_log_dict.values(),
                ["nr34", "nr2", "nr1", "rem", "wake"],
            )
        }
    elif __cm.shape[0] == 4:
        rec_log_dict = {
            "rec_" + ss_label: (confdiag[i][i]) / (ss_dict[i])
            for (ss_label, i) in zip(["nr2", "nr1", "rem", "wake"], range(5))
        }
        pre_log_dict = {
            "pre_"
            + ss_label: (confdiag[i][i]) / (sum(__cm[i]) + confdiag[i][i])
            for (ss_label, i) in zip(["nr2", "nr1", "rem", "wake"], range(5))
        }
        f_m_log_dict = {
            "f_m_" + ss_label: rec * pre * 2 / (rec + pre)
            for (rec, pre, ss_label) in zip(
                rec_log_dict.values(),
                pre_log_dict.values(),
                ["nr2", "nr1", "rem", "wake"],
            )
        }
    rec_df = pd.DataFrame(
        rec_log_dict,
        index=[
            "i",
        ],
    )
    pre_df = pd.DataFrame(
        pre_log_dict,
        index=[
            "i",
        ],
    )
    f_df = pd.DataFrame(
        f_m_log_dict,
        index=[
            "i",
        ],
    )
    output_df = pd.concat([rec_df, pre_df, f_df], axis=0)
    output_path = os.path.join(utils.env.tmp_dir, test_name, "metrics.csv")
    output_df.to_csv(output_path)


if __name__ == "__main__":
    set_seed(0)
    # 環境設定
    CALC_DEVICE = "gpu"
    # CALC_DEVICE = "cpu"
    DEVICE_ID = "0" if CALC_DEVICE == "gpu" else "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DEEETERMINISTIC"] = "1"
    if os.environ["CUDA_VISIBLE_DEVICES"] != "-1":
        tf.keras.backend.set_floatx("float32")
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # tf.config.run_functions_eagerly(True)
    else:
        print("*** cpuで計算します ***")

    # ハイパーパラメータの設定
    # TODO: jsonに移植
    TEST_RUN = True
    EPOCHS = 50
    HAS_ATTENTION = True
    PSE_DATA = False
    HAS_INCEPTION = True
    IS_PREVIOUS = False
    IS_NORMAL = True
    IS_ENN = True  # FIXME: always true so remove here
    IS_MUL_LAYER = False
    CATCH_NREM2 = True
    HAS_DROPOUT = True
    BATCH_SIZE = 64
    N_CLASS = 5
    # KERNEL_SIZE = 256
    KERNEL_SIZE = 128
    STRIDE = 16
    SAMPLE_SIZE = 2000
    UNC_THRETHOLD = 0.3
    DROPOUT_RATE = 0.3
    IS_UNDER_4HZ = False
    DATA_TYPE = "spectrogram"
    FIT_POS = "middle"
    EXPERIMENT_TYPES = (
        "no_cleansing",
        "positive_cleansing",
        "negative_cleansing",
    )
    EXPERIENT_TYPE = "positive_cleansing"
    NORMAL_TAG = "normal" if IS_NORMAL else "sas"
    ATTENTION_TAG = "attention" if HAS_ATTENTION else "no-attention"
    PSE_DATA_TAG = "psedata" if PSE_DATA else "sleepdata"
    INCEPTION_TAG = "inception" if HAS_INCEPTION else "no-inception"
    WANDB_PROJECT = "test" if TEST_RUN else "main_project"
    ENN_TAG = "enn" if IS_ENN else "dnn"
    INCEPTION_TAG += "v2" if IS_MUL_LAYER else ""
    CATCH_NREM2_TAG = "catch_nrem2" if CATCH_NREM2 else "catch_nrem34"
    CLEANSING_TYPE = "no_cleansing"

    # オブジェクトの作成
    pre_process = PreProcess(
        data_type=DATA_TYPE,
        fit_pos=FIT_POS,
        verbose=0,
        kernel_size=KERNEL_SIZE,
        is_previous=IS_PREVIOUS,
        stride=STRIDE,
        is_normal=IS_NORMAL,
        has_nrem2_bias=False,
        has_rem_bias=False,
        model_type=ENN_TAG,
        cleansing_type=CLEANSING_TYPE,
        make_valdata=True,
        has_ignored=True,
        lsp_option="nr2",
    )
    datasets = pre_process.load_sleep_data.load_data(
        load_all=True, pse_data=PSE_DATA
    )

    # 読み込むモデルの日付リストを返す
    MI = pre_process.my_env.mi
    model_date_d = MI.get_ppi()
    model_date_list = model_date_d[CLEANSING_TYPE]

    # モデルのidを記録するためのリスト
    date_id_saving_list = list()

    for (test_id, test_name), date_id in zip(
        enumerate(pre_process.name_list), model_date_list
    ):
        (train, test) = pre_process.split_train_test_from_records(
            datasets, test_id=test_id, pse_data=PSE_DATA
        )

        # 保存用の時間id
        saving_date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        date_id_saving_list.append(saving_date_id)

        # tagの設定
        # TODO: wandb のutilsを作成する
        my_tags = [
            test_name,
            f"kernel:{KERNEL_SIZE}",
            f"stride:{STRIDE}",
            f"sample:{SAMPLE_SIZE}",
            f"model:{ENN_TAG}",
            f"{EXPERIENT_TYPE}",
            f"u_th:{UNC_THRETHOLD}",
        ]
        wandb_config = {
            "test name": test_name,
            "date id": date_id,
            "sample_size": SAMPLE_SIZE,
            "epochs": EPOCHS,
            "kernel": KERNEL_SIZE,
            "stride": STRIDE,
            "fit_pos": FIT_POS,
            "batch_size": BATCH_SIZE,
            "n_class": N_CLASS,
            "experiment": EXPERIENT_TYPE,
        }
        # FIXME: name をコード名にする
        main(
            train=train,
            test=test,
            pre_process=pre_process,
            date_id=date_id,
            pse_data=PSE_DATA,
            test_name=test_name,
            n_class=N_CLASS,
            sample_size=SAMPLE_SIZE,
            batch_size=BATCH_SIZE,
            utils=Utils(
                IS_NORMAL,
                IS_PREVIOUS,
                DATA_TYPE,
                FIT_POS,
                STRIDE,
                KERNEL_SIZE,
                model_type=ENN_TAG,
                cleansing_type=CLEANSING_TYPE,
            ),
            is_under_4hz=IS_UNDER_4HZ,
        )

        # testの時は一人の被験者で止める
        if TEST_RUN:
            print(PyColor.RED_FLASH, "testランのため終了します", PyColor.END)
            break
