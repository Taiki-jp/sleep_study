import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # tensorflow を読み込む前のタイミングですると効果あり
import tensorflow as tf

tf.random.set_seed(0)
from tensorflow.python.keras.engine.training import Model
from data_analysis.utils import Utils
import sys
import datetime
import wandb
from wandb.keras import WandbCallback
from pre_process.pre_process import PreProcess
from nn.model_base import EDLModelBase, edl_classifier_1d
from nn.losses import EDLLoss
from pre_process.json_base import JsonBase
from data_analysis.py_color import PyColor
from collections import Counter


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
    date_id: dict = dict(),
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
    log_all_in_one: bool = False,
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
    ss_train_dict = Counter(y_train)
    ss_test_dict = Counter(y_test)

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

    # NOTE: kernel_size の半分が入力のサイズになる（fft をかけているため）
    if data_type == "spectrum":
        shape = (int(kernel_size / 2), 1)
    elif data_type == "spectrogram":
        shape = (128, 512, 1)
    else:
        print("correct here based on your model")
        sys.exit(1)

    def _load_model(
        is_positive: bool = False, is_negative: bool = False
    ) -> Model:
        print(PyColor.GREEN, f"*** {test_name}のモデルを読み込みます ***", PyColor.END)
        if is_positive and is_negative == False:
            path = os.path.join(
                os.environ["sleep"],
                "models",
                test_name,
                date_id["positive"],
            )
        elif is_negative and is_positive == False:
            path = os.path.join(
                os.environ["sleep"],
                "models",
                test_name,
                date_id["negative"],
            )
        else:
            path = os.path.join(
                os.environ["sleep"],
                "models",
                test_name,
                date_id["nothing"],
            )

        # path があっているか確認
        if not os.path.exists(path):
            print(PyColor.RED_FLASH, f"{path}は存在しません", PyColor.END)
            sys.exit(1)
        model = tf.keras.models.load_model(
            path, custom_objects={"EDLLoss": EDLLoss(K=n_class, annealing=0.1)}
        )
        print(PyColor.GREEN, f"*** {test_name}のモデルを読み込みました ***", PyColor.END)
        return model

    # データクレンジングを行うベースとなるモデルを読み込む
    model = _load_model(is_negative=False, is_positive=False)
    # ポジティブクレンジングを行ったモデルを読み込む
    positive_model = _load_model(is_negative=False, is_positive=True)
    # NOTE : そのためone-hotの状態でデータを読み込む必要がある
    x, y = (x_train, y_train)

    def _sep_unc_data(x, y) -> tuple:
        evidence = model.predict(x, batch_size=batch_size)
        alpha = evidence + 1
        unc = n_class / tf.reduce_sum(alpha, axis=1, keepdims=True)
        mask_under_threthold = unc <= unc_threthold
        mask_over_threthold = unc >= unc_threthold

        return (
            tf.boolean_mask(
                x, mask_under_threthold.numpy().reshape(x.shape[0])
            ),
            tf.boolean_mask(
                y, mask_under_threthold.numpy().reshape(x.shape[0])
            ),
        ), (
            tf.boolean_mask(
                x, mask_over_threthold.numpy().reshape(x.shape[0])
            ),
            tf.boolean_mask(
                y, mask_over_threthold.numpy().reshape(x.shape[0])
            ),
        )

    (_x, _y), (_x_over_threthold, _y_over_threthold) = _sep_unc_data(x=x, y=y)
    (_x_test, _y_test), (
        _x_test_over_threthold,
        _y_test_over_threthold,
    ) = _sep_unc_data(x=x_test, y=y_test)

    # データが拾えなかった場合は終了
    if _x.shape[0] == 0 or _x_test.shape[0] == 0:
        return

    # train_or_test = ("train", "test")
    train_or_test = "test"
    # base_model_or_positive_model = (model, positive_model)
    datas = (
        # ((_x, _y), (_x_over_threthold, _y_over_threthold)),
        ((_x_test, _y_test), (_x_test_over_threthold, _y_test_over_threthold)),
    )

    for is_train_or_test_label, __datas in zip(train_or_test, datas):
        evidence_base = model.predict(__datas[0][0], batch_size=batch_size)
        evidence_positive = positive_model.predict(__datas[1][0])
        evidence = tf.concat([evidence_base, evidence_positive], axis=0)
        __y = tf.concat([__datas[0][1], __datas[1][1]], axis=0)
        # 混合行列をwandbに送信
        # utils.conf_mat2Wandb(
        #     y=__y.numpy(),
        #     evidence=evidence,
        #     train_or_test=is_train_or_test_label,
        #     test_label=test_name,
        #     date_id=saving_date_id,
        # )
        # for is_separating in (True, False):

        #     # # 不確かさのヒストグラムをwandbに送信 NOTE: separate_each_ss を Ttrue にすると睡眠段階のヒストグラムになる
        #     utils.u_hist2Wandb(
        #         y=__y.numpy(),
        #         evidence=evidence,
        #         train_or_test=is_train_or_test_label,
        #         test_label=test_name,
        #         date_id=saving_date_id,
        #         separate_each_ss=is_separating,
        #     )
        # # 閾値を設定して分類した時の一致率とサンプル数をwandbに送信
        utils.u_threshold_and_acc2Wandb(
            y=__y.numpy(),
            evidence=evidence,
            train_or_test=is_train_or_test_label,
            test_label=test_name,
            date_id=saving_date_id,
            log_all_in_one=log_all_in_one,
        )
    # wandb終了
    wandb.finish()


if __name__ == "__main__":
    # 環境設定
    CALC_DEVICE = "gpu"
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
    TEST_RUN = False
    WANDB_PROJECT = "データマージの表示テスト00"
    HAS_ATTENTION = True
    PSE_DATA = False
    HAS_INCEPTION = True
    IS_PREVIOUS = False
    IS_NORMAL = True
    IS_ENN = True  # FIXME: always true so remove here
    IS_MUL_LAYER = False
    CATCH_NREM2 = True
    EPOCHS = 200
    BATCH_SIZE = 256
    N_CLASS = 5
    KERNEL_SIZE = 512
    STRIDE = 1024
    SAMPLE_SIZE = 5000
    UNC_THRETHOLD = 0.5
    DATA_TYPE = "spectrum"
    FIT_POS = "middle"
    EXPERIMENT_TYPES = (
        "no_cleansing",
        "positive_cleansing",
        "negative_cleansing",
    )
    EXPERIENT_TYPE = "positive_cleansing"  # ここで model
    NORMAL_TAG = "normal" if IS_NORMAL else "sas"
    ATTENTION_TAG = "attention" if HAS_ATTENTION else "no-attention"
    PSE_DATA_TAG = "psedata" if PSE_DATA else "sleepdata"
    INCEPTION_TAG = "inception" if HAS_INCEPTION else "no-inception"
    # WANDB_PROJECT = "data_selecting_test" if TEST_RUN else "data_selecting_0831"
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
    ][f"kernel_{str(KERNEL_SIZE)}"]["no_cleansing"]

    date_id_list_positive = JB.json_dict[ENN_TAG][DATA_TYPE][FIT_POS][
        f"stride_{str(STRIDE)}"
    ][f"kernel_{str(KERNEL_SIZE)}"]["positive_cleansing"]

    date_id_list_negative = JB.json_dict[ENN_TAG][DATA_TYPE][FIT_POS][
        f"stride_{str(STRIDE)}"
    ][f"kernel_{str(KERNEL_SIZE)}"]["negative_cleansing"]

    date_id_keys = ("nothing", "negative", "positive")
    date_id_values = [
        date_id_list,
        date_id_list_negative,
        date_id_list_positive,
    ]

    # 辞書をリスト型で保持
    mapped = map(
        lambda x, y, z: dict(nothing=x, negative=y, positive=z),
        date_id_list,
        date_id_list_negative,
        date_id_list_positive,
    )
    model_date_list = list(mapped)

    # モデルのidを記録するためのリスト
    date_id_saving_list = list()

    for test_id, (test_name, date_id) in enumerate(
        zip(pre_process.name_list, model_date_list)
    ):
        (train, test) = pre_process.split_train_test_from_records(
            datasets, test_id=test_id, pse_data=PSE_DATA
        )

        # 保存用の時間id
        saving_date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        date_id_saving_list.append(saving_date_id)

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
            EXPERIENT_TYPE,
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
            batch_size=BATCH_SIZE,
            unc_threthold=UNC_THRETHOLD,
            experiment_type=EXPERIENT_TYPE,
            epochs=EPOCHS,
            saving_date_id=saving_date_id,
            log_all_in_one=True,
        )

        # testの時は一人の被験者で止める
        if TEST_RUN:
            break
