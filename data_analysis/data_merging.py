import datetime
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops.numpy_ops.np_math_ops import positive

# import wandb
from data_analysis.py_color import PyColor
from data_analysis.utils import Utils

# from mywandb.utils import make_ss_dict4wandb
from nn.utils import load_model, separate_unc_data
from pre_process.json_base import JsonBase
from pre_process.pre_process import PreProcess


def main(
    name: str,
    # project: str,
    train: list,
    test: list,
    pre_process: PreProcess,
    utils: Utils,
    my_tags: list = None,
    n_class: int = 5,
    test_name: str = None,
    date_id: dict = dict(),
    sample_size: int = 0,
    # wandb_config: dict = dict(),
    batch_size: int = 0,
    unc_threthold: float = 0,
    saving_date_id: str = "",
    log_all_in_one: bool = False,
    is_under_4hz: bool = False,
    pse_data: bool = False,
):

    # データセットの作成
    ((_), (_), (x_test, y_test),) = pre_process.make_dataset(
        train=train,
        test=test,
        is_storchastic=False,
        pse_data=pse_data,
        to_one_hot_vector=False,
        each_data_size=sample_size,
        is_under_4hz=is_under_4hz,
    )

    # config の追加
    # wandb_config.update(make_ss_dict4wandb(ss_array=y_test[0], is_train=False))

    # wandbの初期化
    # wandb.init(
    #     name=name,
    #     project=project,
    #     tags=my_tags,
    #     config=wandb_config,
    #     sync_tensorboard=True,
    #     dir=pre_process.my_env.project_dir,
    # )

    # データクレンジングを行うベースとなるモデルを読み込む
    model = load_model(
        loaded_name=test_name,
        n_class=n_class,
        verbose=1,
        model_id=date_id,
        is_positive=False,
        is_negative=False,
    )
    # データがキャッチできていない場合はreturn
    if model is None:
        print(PyColor.RED_FLASH, "modelが空です", PyColor.END)
        sys.exit(1)
    # ポジティブクレンジングを行ったモデルを読み込む
    positive_model = load_model(
        loaded_name=test_name,
        n_class=n_class,
        verbose=1,
        model_id=date_id,
        is_positive=True,
        is_negative=False,
    )
    # データがキャッチできていない場合はreturn
    if positive_model is None:
        print(PyColor.RED_FLASH, "modelが空です", PyColor.END)
        return

    # テストデータのクレンジング
    (_x_test, _y_test) = separate_unc_data(
        x=x_test,
        y=y_test[0],
        model=model,
        batch_size=batch_size,
        n_class=n_class,
        experiment_type="positive_cleansing",
        unc_threthold=unc_threthold,
        verbose=0,
    )

    # # ベースモデルの不確実なデータセットに対する一致率を計算しwandbに送信
    # acc_of_unc_high_data = list()
    # for base_or_positive, _model in zip(
    #     ("base", "positive"), (model, positive_model)
    # ):
    #     acc_of_unc_high_data.append(
    #         utils.calc_ss_acc(
    #             x=_x_test,
    #             y=_y_test,
    #             model=_model,
    #             n_class=n_class,
    #             batch_size=batch_size,
    #             base_or_positive=base_or_positive,
    #             log2wandb=False,
    #         )
    #     )

    # # クレンジング後のデータに対してグラフを作成
    # 不確実性の高いデータのみで一致率を計算
    evidence_base = model.predict(_x_test, batch_size=batch_size)
    evidence_positive = positive_model.predict(_x_test)

    # 睡眠段階の予測
    (
        evi_base,
        alp_base,
        unc_base,
        y_pred_base,
    ) = utils.calc_enn_output_from_evidence(evidence=evidence_base)
    (
        evi_pos,
        alp_pos,
        unc_pos,
        y_pred_pos,
    ) = utils.calc_enn_output_from_evidence(evidence=evidence_positive)
    # 一致率の計算
    acc_base = utils.calc_acc_from_pred(
        y_true=_y_test.numpy(),
        y_pred=y_pred_base,
        log_label="base",
        log2wandb=False,
    )
    acc_sub = utils.calc_acc_from_pred(
        y_true=_y_test.numpy(),
        y_pred=y_pred_pos,
        log_label="sub",
        log2wandb=False,
    )
    # 一致率のcsv出力
    output_path = os.path.join(
        utils.env.tmp_dir, test_name, "acc_of_high_unc_datas.csv"
    )
    df_acc = pd.DataFrame(
        np.array([acc_base, acc_sub]).reshape(1, 2),
        columns=["acc_base", "acc_sub"],
    )
    df_acc.to_csv(output_path)
    # 睡眠段階，不確実性のcsv出力
    output_path = os.path.join(utils.env.tmp_dir, test_name, "ss_and_unc.csv")
    df_result = pd.DataFrame(
        np.array(
            [_y_test.numpy(), y_pred_base, y_pred_pos, unc_base, unc_pos]
        ).T,
        columns=[
            "y_true",
            "base_pred",
            "positive_pred",
            "unc_base",
            "unc_pos",
        ],
    )
    df_result.to_csv(output_path)
    # 睡眠段階の比較
    # utils.compare_ss(y_true=y_test[0], y_pred=y_pred_pos, test_name=test_name)
    # wandb終了
    # wandb.finish()


if __name__ == "__main__":
    # 環境設定
    CALC_DEVICE = "gpu"
    DEVICE_ID = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID
    tf.keras.backend.set_floatx("float32")
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # ANCHOR: ハイパーパラメータの設定
    TEST_RUN = False
    # WANDB_PROJECT = "test" if TEST_RUN else "20220121_nidan_merge"
    IS_MUL_LAYER = False
    IS_NORMAL = True
    IS_ENN = False
    ENN_TAG = "enn" if IS_ENN else "dnn"
    CATCH_NREM2 = True
    BATCH_SIZE = 512
    N_CLASS = 5
    STRIDE = 16
    KERNEL_SIZE = 128
    SAMPLE_SIZE = 2000
    UNC_THRETHOLD = 0.3
    PSE_DATA = False
    CLEANSING_TYPE = "no_cleansing"
    EXPERIMENT_TYPES = (
        "no_cleansing",
        "positive_cleansing",
        "negative_cleansing",
    )
    DATA_TYPE = "spectrogram"
    FIT_POS = "middle"
    IS_PREVIOUS = False

    # オブジェクトの作成
    pre_process = PreProcess(
        # data_type=DATA_TYPE,
        # fit_pos=FIT_POS,
        # verbose=0,
        # kernel_size=KERNEL_SIZE,
        # is_previous=IS_PREVIOUS,
        # stride=STRIDE,
        # is_normal=True,
        # cleansing_type=CLEANSING_TYPE,
        data_type=DATA_TYPE,
        fit_pos=FIT_POS,
        verbose=0,
        kernel_size=KERNEL_SIZE,
        is_previous=IS_PREVIOUS,
        stride=STRIDE,
        is_normal=True,
        has_nrem2_bias=True,
        has_rem_bias=False,
        model_type="enn",
        cleansing_type=CLEANSING_TYPE,
        make_valdata=True,
        has_ignored=True,
        lsp_option="nr2",
    )

    # 読み込むモデルの日付リストを返す
    MI = pre_process.my_env.mi
    datasets = pre_process.load_sleep_data.load_data(
        load_all=True, pse_data=PSE_DATA
    )

    model_date_d = MI.get_ppi()
    model_date_list = [
        {"nothing": _noth_list, "positive": _pos_list}
        for (_noth_list, _pos_list) in zip(
            model_date_d["no_cleansing"], model_date_d["positive_cleansing"]
        )
    ]
    try:
        assert len(model_date_list) == len(model_date_d)
    except AssertionError as AE:
        print(AE)

    # モデルのidを記録するためのリスト
    date_id_saving_list: List[str] = list()

    for test_id, (test_name, date_id) in enumerate(
        zip(pre_process.name_list, model_date_list)
    ):
        (train, test) = pre_process.split_train_test_from_records(
            datasets, test_id=test_id
        )

        # 保存用の時間id
        saving_date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # date_id_saving_list.append(saving_date_id)

        # tagの設定
        my_tags = [
            test_name,
        ]
        # wandb_config = {
        #     "test name": test_name,
        # }
        # FIXME: name をコード名にする
        main(
            name=f"{test_name}",
            # project=WANDB_PROJECT,
            train=train,
            test=test,
            pre_process=pre_process,
            my_tags=my_tags,
            date_id=date_id,
            test_name=test_name,
            n_class=N_CLASS,
            # wandb_config=wandb_config,
            batch_size=BATCH_SIZE,
            unc_threthold=UNC_THRETHOLD,
            saving_date_id=saving_date_id,
            log_all_in_one=True,
            sample_size=SAMPLE_SIZE,  # これがないとtrainの分割のところでデータがなくてエラーが起きてしまう
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
        )

        # testの時は一人の被験者で止める
        if TEST_RUN:
            break
