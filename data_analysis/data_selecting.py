import os
from typing import Dict, Tuple

from nn.wandb_classification_callback import WandbClassificationCallback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # tensorflow を読み込む前のタイミングですると効果あり
import tensorflow as tf

tf.random.set_seed(0)
import datetime
import sys
from collections import Counter

import wandb
from data_analysis.py_color import PyColor
from data_analysis.utils import Utils
from mywandb.utils import make_ss_dict4wandb
from nn.losses import EDLLoss
from nn.model_base import EDLModelBase, edl_classifier_1d
from nn.utils import load_model, separate_unc_data
from pre_process.json_base import JsonBase
from pre_process.pre_process import PreProcess
from wandb.keras import WandbCallback


# TODO: 使っていない引数の削除
def main(
    name: str,
    project: str,
    train: list,
    test: list,
    pre_process: PreProcess,
    my_tags: list = None,
    n_class: int = 5,
    pse_data: bool = False,
    test_name: str = None,
    date_id: dict = None,
    has_attention: bool = False,
    has_inception: bool = True,
    data_type: str = None,
    sample_size: int = 0,
    is_enn: bool = True,
    wandb_config: dict = dict(),
    kernel_size: int = 0,
    is_mul_layer: bool = False,
    batch_size: int = 0,
    unc_threthold: float = 0,
    epochs: int = 1,
    experiment_type: str = "",
    saving_date_id: str = "",
    has_dropout: bool = False,
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
    ss_train_dict: Dict[int, int] = Counter(y_train)
    ss_test_dict: Dict[int, int] = Counter(y_test)

    # config の追加
    added_config = {
        "attention": has_attention,
        "inception": has_inception,
        "test wake before replaced": ss_test_dict[4],
        "test rem before replaced": ss_test_dict[3],
        "test nr1 before replaced": ss_test_dict[2],
        "test nr2 before replaced": ss_test_dict[1],
        "test nr34 before replaced": ss_test_dict[0],
        "train wake before replaced": ss_train_dict[4],
        "train rem before replaced": ss_train_dict[3],
        "train nr1 before replaced": ss_train_dict[2],
        "train nr2 before replaced": ss_train_dict[1],
        "train nr34 before replaced": ss_train_dict[0],
    }
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

    # NOTE: earernel_size の半分が入力のサイズになる（fft をかけているため）
    # TODO: mypyでshapeが違うとエラーになる
    if data_type == "spectrum":
        shape: Tuple[int, int] = (int(kernel_size / 2), 1)
    elif data_type == "spectrogram":
        shape: Tuple[int, int, int] = (128, 512, 1)
    else:
        print("correct here based on your model")
        sys.exit(1)

    inputs = tf.keras.Input(shape=shape)
    outputs = edl_classifier_1d(
        x=inputs,
        n_class=n_class,
        has_attention=has_attention,
        has_inception=has_inception,
        is_mul_layer=is_mul_layer,
        has_dropout=has_dropout,
    )

    model = load_model(
        loaded_name=test_name, model_id=date_id, n_class=n_class, verbose=0
    )
    # NOTE : そのためone-hotの状態でデータを読み込む必要がある
    # TODO: このコピーいる？
    x, y = (x_train, y_train)

    # 訓練データのクレンジング
    (_x, _y) = separate_unc_data(
        x=x,
        y=y,
        model=model,
        batch_size=batch_size,
        n_class=n_class,
        experiment_type=experiment_type,
        unc_threthold=unc_threthold,
        verbose=0,
    )
    # テストデータのクレンジング
    (_x_test, _y_test) = separate_unc_data(
        x=x_test,
        y=y_test,
        model=model,
        batch_size=batch_size,
        n_class=n_class,
        experiment_type=experiment_type,
        unc_threthold=unc_threthold,
        verbose=0,
    )

    # データクレンジングされた後のデータ数をログにとる
    wandb.log(make_ss_dict4wandb(_y, is_train=True), commit=False)
    wandb.log(make_ss_dict4wandb(_y_test, is_train=False), commit=False)

    # データが拾えなかった場合は終了
    catching_flag_train = utils.stop_early(y=_y, mode="catching_assertion")
    catching_flag_test = utils.stop_early(y=_y_test, mode="catching_assertion")
    # 早期終了フラグが立った場合はそこで終了
    if catching_flag_train or catching_flag_test:
        return
    # if _x.shape[0] == 0 or _x_test.shape[0] == 0:
    #     return

    _model = EDLModelBase(inputs=inputs, outputs=outputs)
    _model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=EDLLoss(K=n_class, annealing=0.1),
        metrics=[
            "accuracy",
        ],
    )
    log_dir = os.path.join(
        pre_process.my_env.project_dir, "my_edl", test_name, date_id["nothing"]
    )
    tf_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    _model.fit(
        x=_x,
        y=_y,
        batch_size=batch_size,
        validation_data=(_x_test, _y_test),
        epochs=epochs,
        callbacks=[
            tf_callback,
            WandbClassificationCallback(
                validation_data=(_x_test, _y_test),
                log_confusion_matrix=True,
                labels=["nr34", "nr2", "nr1", "rem", "wake"],
            ),
        ],
        verbose=2,
    )

    # 混合行列・不確かさ・ヒストグラムの作成
    tuple_x = (_x, _x_test)
    tuple_y = (_y, _y_test)
    # 混合行列をwandbに送信
    for train_or_test, __x, __y in zip(["train", "test"], tuple_x, tuple_y):
        evidence = _model.predict(__x)
        utils.make_graphs(
            y=__y.numpy(),
            evidence=evidence,
            train_or_test=train_or_test,
            graph_person_id=test_name,
            calling_graph="all",
            graph_date_id=saving_date_id,
            is_each_unc=True,
        )

    # モデルの保存
    path = os.path.join(
        pre_process.my_env.models_dir, test_name, saving_date_id
    )
    _model.save(path)
    # wandb終了
    wandb.finish()


if __name__ == "__main__":
    # 環境設定
    CALC_DEVICE = "gpu"
    # CALC_DEVICE = "cpu"
    DEVICE_ID = "0" if CALC_DEVICE == "gpu" else "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID
    if os.environ["CUDA_VISIBLE_DEVICES"] != "-1":
        tf.keras.backend.set_floatx("float32")
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # tf.config.run_functions_eagerly(True)
    else:
        print("*** cpuで計算します ***")
        # なんか下のやつ使えなくなっている、、
        # tf.config.run_functions_eagerly(True)

    # ハイパーパラメータの設定
    # TODO: jsonに移植
    TEST_RUN = False
    HAS_ATTENTION = True
    PSE_DATA = False
    HAS_INCEPTION = True
    IS_PREVIOUS = False
    IS_NORMAL = True
    IS_ENN = True  # FIXME: always true so remove here
    IS_MUL_LAYER = False
    CATCH_NREM2 = True
    HAS_DROPOUT = True
    EPOCHS = 100
    BATCH_SIZE = 128
    N_CLASS = 5
    KERNEL_SIZE = 512
    STRIDE = 480
    SAMPLE_SIZE = 10000
    UNC_THRETHOLD = 0.5
    DATA_TYPE = "spectrum"
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
    WANDB_PROJECT = "test" if TEST_RUN else "positive_cleansing"
    ENN_TAG = "enn" if IS_ENN else "dnn"
    INCEPTION_TAG += "v2" if IS_MUL_LAYER else ""
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
        has_nrem2_bias=True,
    )
    datasets = pre_process.load_sleep_data.load_data(
        load_all=True, pse_data=PSE_DATA
    )
    utils = Utils(catch_nrem2=CATCH_NREM2)

    # 読み込むモデルの日付リストを返す
    JB = JsonBase("../nn/model_id.json")
    JB.load()
    model_date_list = JB.make_list_of_dict_from_mul_list(
        "enn", "spectrum", "middle", "stride_480", "kernel_512"
    )
    try:
        assert len(model_date_list) != 0
    except AssertionError("model_date_listの中身が空です"):
        sys.exit(1)

    # モデルのidを記録するためのリスト
    date_id_saving_list = list()
    subjects_id = [i for i in range(len(pre_process.name_list))]

    for test_id, test_name, date_id in zip(
        subjects_id[47:], pre_process.name_list[47:], model_date_list[47:]
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
            name=f"{test_name}",
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
            batch_size=BATCH_SIZE,
            unc_threthold=UNC_THRETHOLD,
            experiment_type=EXPERIENT_TYPE,
            epochs=EPOCHS,
            saving_date_id=saving_date_id,
            has_dropout=HAS_DROPOUT,
        )

        # testの時は一人の被験者で止める
        if TEST_RUN:
            break

    # モデルを保存しないのでコメントアウト
    # TODO: モデルの保存を行う変数に応じて場合分け
    JB.dump(
        keys=[
            ENN_TAG,
            DATA_TYPE,
            FIT_POS,
            f"stride_{str(STRIDE)}",
            f"kernel_{str(KERNEL_SIZE)}",
            f"{EXPERIENT_TYPE}",
        ],
        value=date_id_saving_list,
    )
