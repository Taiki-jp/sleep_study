from wandb.keras import WandbCallback
from pre_process.pre_process import PreProcess
import copy
from random import sample
import sys
from data_analysis.py_color import PyColor
import os
import datetime
import wandb
import tensorflow as tf
from collections import Counter
from pre_process.pre_process import PreProcess
from nn.model_base import classifier4enn, edl_classifier_1d, spectrum_conv
from nn.losses import EDLLoss
from pre_process.json_base import JsonBase
import numpy as np
from data_analysis.utils import Utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # tensorflow を読み込む前のタイミングですると効果あり


def main(
    name,
    project,
    train,
    test,
    pre_process,
    my_tags=None,
    n_class=5,
    pse_data=False,
    test_name=None,
    date_id=None,
    has_attention=False,
    has_inception=True,
    data_type=None,
    sample_size=0,
    is_enn=True,
    wandb_config=dict(),
    kernel_size=0,
    is_mul_layer=False,
):

    # データセットの作成
    (x_train, y_train), (x_test, y_test) = pre_process.make_dataset(
        train=train,
        test=test,
        is_storchastic=False,
        pse_data=pse_data,
        to_one_hot_vector=False,
        each_data_size=sample_size,
    )
    # データセットの数を表示
    print(f"training data : {x_train.shape}")

    # config の追加
    added_config = {"attention": has_attention, "inception": has_inception}
    wandb_config.update(added_config)

    # wandbの初期化
    wandb.init(
        name=name,
        project=project,
        tags=my_tags,
        config=wandb_config,
        sync_tensorboard=True,
        dir=pre_process.my_env.project_dir,
    )

    # モデルの読み込み（コンパイル済み）
    print(f"*** {test_name}のモデルを読み込みます ***")
    exploit_path = os.path.join(
        os.environ["sleep"], "models", test_name, "exploit", "20210716-092851"
    )
    main_path = os.path.join(
        os.environ["sleep"], "models", test_name, "main", "20210716-092851"
    )
    sub_path = os.path.join(
        os.environ["sleep"], "models", test_name, "sub", "20210716-092851"
    )

    exploit_model = tf.keras.models.load_model(
        exploit_path, custom_objects={"EDLLoss": EDLLoss(K=5, annealing=0.1)}
    )
    main_model = tf.keras.models.load_model(
        main_path, custom_objects={"EDLLoss": EDLLoss(K=5, annealing=0.1)}
    )
    sub_model = tf.keras.models.load_model(
        sub_path, custom_objects={"EDLLoss": EDLLoss(K=5, annealing=0.1)}
    )
    print(f"*** {test_name}のモデルを読み込みました ***")

    # モデルの評価（どの関数が走る？ => lossのcallが呼ばれてる）
    # NOTE : そのためone-hotの状態でデータを読み込む必要がある

    # trainとtestのループ処理
    train_test_holder = [(x_train, y_train), (x_test, y_test)]
    train_test_label = ["train", "test"]
    for train_or_test, data in zip(train_test_label, train_test_holder):
        x, y = data
        # EDLBase.__call__が走る
        hidden = exploit_model.predict(x, batch_size=32)
        evidence_main = main_model.predict(hidden, batch_size=32)
        evidence_sub = sub_model.predict(hidden, batch_size=32)
        alpha_main = evidence_main + 1
        unc_main = n_class / tf.reduce_sum(alpha_main, axis=1, keepdims=True)
        evidence_merged = (
            1 - unc_main
        ) * evidence_main + unc_main * evidence_sub
        # 混合行列をwandbに送信
        utils.conf_mat2Wandb(
            y=y,
            evidence=evidence_main,
            train_or_test=train_or_test,
            test_label=test_name,
            date_id=date_id,
        )
        # 不確かさのヒストグラムをwandbに送信
        utils.u_hist2Wandb(
            y=y,
            evidence=evidence_main,
            train_or_test=train_or_test,
            test_label=test_name,
            date_id=date_id,
            separate_each_ss=False,
            unc=unc_main,
        )
        # 閾値を設定して分類した時の一致率とサンプル数をwandbに送信
        utils.u_threshold_and_acc2Wandb(
            y=y,
            evidence=evidence_main,
            train_or_test=train_or_test,
            test_label=test_name,
            date_id=date_id,
        )
        # 先にwandbが閉じないように10秒待つ
        # time.sleep(10)
    # wandb終了
    wandb.finish()


if __name__ == "__main__":
    # 環境設定
    try:
        tf.keras.backend.set_floatx("float32")
        tf.config.run_functions_eagerly(True)
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        print("*** cpuで計算します ***")

    # ハイパーパラメータの設定
    TEST_RUN = False
    HAS_ATTENTION = True
    PSE_DATA = False
    HAS_INCEPTION = True
    IS_PREVIOUS = False
    IS_NORMAL = True
    IS_ENN = True
    IS_MUL_LAYER = False
    CATCH_NREM2 = True
    EPOCHS = 20
    BATCH_SIZE = 32
    N_CLASS = 5
    KERNEL_SIZE = 512
    STRIDE = 16
    SAMPLE_SIZE = 5000
    DATA_TYPE = "spectrum"
    FIT_POS = "middle"
    NORMAL_TAG = "normal" if IS_NORMAL else "sas"
    ATTENTION_TAG = "attention" if HAS_ATTENTION else "no-attention"
    PSE_DATA_TAG = "psedata" if PSE_DATA else "sleepdata"
    INCEPTION_TAG = "inception" if HAS_INCEPTION else "no-inception"
    WANDB_PROJECT = "test" if TEST_RUN else "edl-analysis"
    ENN_TAG = "enn" if IS_ENN else "dnn"
    INCEPTION_TAG += "v2" if IS_MUL_LAYER else INCEPTION_TAG
    CATCH_NREM2_TAG = "catch_nrem2" if CATCH_NREM2 else "catch_nrem34"

    # オブジェクトの作成
    pre_process = PreProcess(
        data_type=DATA_TYPE,
        fit_pos=FIT_POS,
        verbose=0,
        kernel_size=KERNEL_SIZE,
        is_previous=IS_PREVIOUS,
        stride=STRIDE,
        is_normal=IS_NORMAL,
    )
    datasets = pre_process.load_sleep_data.load_data(
        load_all=True, pse_data=PSE_DATA
    )
    utils = Utils(catch_nrem2=CATCH_NREM2)

    # 読み込むモデルの日付リストを返す
    JB = JsonBase("../nn/model_id.json")
    JB.load()
    date_id_list = JB.json_dict[ENN_TAG][DATA_TYPE][FIT_POS][
        f"stride_{str(STRIDE)}"
    ][f"kernel_{str(KERNEL_SIZE)}"]

    for test_id, (test_name, date_id) in enumerate(
        zip(["140711_Yamamoto"], ["20210716-092851"])
    ):
        (train, test) = pre_process.split_train_test_from_records(
            datasets, test_id=test_id, pse_data=PSE_DATA
        )
        # tagの設定
        my_tags = [
            test_name,
            PSE_DATA_TAG,
            ATTENTION_TAG,
            INCEPTION_TAG,
            DATA_TYPE,
            FIT_POS,
            f"kernel_{KERNEL_SIZE}",
            f"stride_{STRIDE}",
            f"sample_{SAMPLE_SIZE}",
            ENN_TAG,
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
        }
        main(
            name=f"edl-{test_name}",
            project=WANDB_PROJECT,
            train=train,
            test=test,
            pre_process=pre_process,
            my_tags=my_tags,
            has_attention=HAS_ATTENTION,
            date_id=date_id,
            pse_data=PSE_DATA,
            test_name=test_name,
            has_inception=HAS_INCEPTION,
            n_class=N_CLASS,
            data_type=DATA_TYPE,
            sample_size=SAMPLE_SIZE,
            is_enn=IS_ENN,
            wandb_config=wandb_config,
            kernel_size=KERNEL_SIZE,
            is_mul_layer=IS_MUL_LAYER,
        )

        # testの時は一人の被験者で止める
        if TEST_RUN:
            break
