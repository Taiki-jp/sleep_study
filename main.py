from nn.wandb_classification_callback import WandbClassificationCallback
import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.random.set_seed(100)
import datetime
from tensorflow.python.framework.ops import Tensor
import wandb
from wandb.keras import WandbCallback
from pre_process.pre_process import PreProcess
from nn.model_base import EDLModelBase, edl_classifier_1d
from nn.losses import EDLLoss

# from nn.metrics import CategoricalTruePositives
from pre_process.json_base import JsonBase
from data_analysis.py_color import PyColor
from pre_process.record import Record
from collections import Counter


def main(
    name: str,
    project: str,
    train: list,
    test: list,
    pre_process: PreProcess,
    epochs: int = 1,
    save_model: bool = False,
    my_tags: list = None,
    batch_size: int = 32,
    n_class: int = 5,
    pse_data: bool = False,
    test_name: str = None,
    date_id: str = None,
    has_attention: bool = False,
    has_inception: bool = True,
    data_type: str = None,
    sample_size: int = 0,
    is_enn: bool = True,
    wandb_config: dict = dict(),
    kernel_size: int = 0,
    is_mul_layer: bool = False,
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

    # モデルの作成とコンパイル
    # NOTE: kernel_size の半分が入力のサイズになる（fft をかけているため）
    if data_type == "spectrum":
        shape = (int(kernel_size / 2), 1)
    elif data_type == "spectrogram":
        shape = (128, 512, 1)
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
    )
    if is_enn:
        model = EDLModelBase(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=EDLLoss(K=n_class, annealing=0.1),
            metrics=[
                "accuracy",
                # CategoricalTruePositives(
                #     target_class=0, data_size=sample_size * n_class, n_class=5
                # ),
                # CategoricalTruePositives(
                #     target_class=1, data_size=sample_size * n_class, n_class=5
                # ),
                # CategoricalTruePositives(
                #     target_class=2, data_size=sample_size * n_class, n_class=5
                # ),
                # CategoricalTruePositives(
                #     target_class=3, data_size=sample_size * n_class, n_class=5
                # ),
                # CategoricalTruePositives(
                #     target_class=4, data_size=sample_size * n_class, n_class=5
                # ),
                # CategoricalTruePositives(),
            ],
        )

    else:
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=[
                "accuracy",
                # CategoricalTruePositives(
                #     target_class=0, data_size=sample_size * n_class, n_class=5
                # ),
                # CategoricalTruePositives(
                #     target_class=1, data_size=sample_size * n_class, n_class=5
                # ),
                # CategoricalTruePositives(
                #     target_class=2, data_size=sample_size * n_class, n_class=5
                # ),
                # CategoricalTruePositives(
                #     target_class=3, data_size=sample_size * n_class, n_class=5
                # ),
                # CategoricalTruePositives(
                #     target_class=4, data_size=sample_size * n_class, n_class=5
                # ),
            ],
        )

    # tensorboard作成
    log_dir = os.path.join(
        pre_process.my_env.project_dir, "my_edl", test_name, date_id
    )
    tf_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    # TODO: 使いたいけど、何をもとに早期終了するかは難しい、、
    # early_callback = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss',
    # )

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        epochs=epochs,
        callbacks=[
            tf_callback,
            # WandbCallback(),
            WandbClassificationCallback(
                validation_data=(x_test, y_test),
                log_confusion_matrix=True,
                labels=["nr34", "nr2", "nr1", "rem", "wake"],
            ),
        ],
        verbose=2,
    )

    if save_model:
        print(PyColor().GREEN_FLASH, "モデルを保存します ...", PyColor().END)
        path = os.path.join(pre_process.my_env.models_dir, test_name, date_id)
        model.save(path)
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
    TEST_RUN = True
    HAS_ATTENTION = True
    PSE_DATA = False
    HAS_INCEPTION = True
    IS_PREVIOUS = False
    IS_NORMAL = True
    IS_ENN = True
    # FIXME: 多層化はとりあえずいらない
    IS_MUL_LAYER = False
    HAS_NREM2_BIAS = False
    EPOCHS = 100
    BATCH_SIZE = 512
    N_CLASS = 5
    KERNEL_SIZE = 512
    STRIDE = 1024
    SAMPLE_SIZE = 5000
    DATA_TYPE = "spectrum"
    FIT_POS = "middle"
    NORMAL_TAG = "normal" if IS_NORMAL else "sas"
    ATTENTION_TAG = "attention" if HAS_ATTENTION else "no-attention"
    PSE_DATA_TAG = "psedata" if PSE_DATA else "sleepdata"
    INCEPTION_TAG = "inception" if HAS_INCEPTION else "no-inception"
    WANDB_PROJECT = "test" if TEST_RUN else "master"
    ENN_TAG = "enn" if IS_ENN else "dnn"
    INCEPTION_TAG += "v2" if IS_MUL_LAYER else ""

    # 記録用のjsonファイルを読み込む
    JB = JsonBase("../nn/model_id.json")
    JB.load()
    # オブジェクトの作成
    pre_process = PreProcess(
        data_type=DATA_TYPE,
        fit_pos=FIT_POS,
        verbose=0,
        kernel_size=KERNEL_SIZE,
        is_previous=IS_PREVIOUS,
        stride=STRIDE,
        is_normal=IS_NORMAL,
        has_nrem2_bias=HAS_NREM2_BIAS,
    )
    datasets = pre_process.load_sleep_data.load_data(
        load_all=True,
        pse_data=PSE_DATA,
    )
    # モデルのidを記録するためのリスト
    date_id_saving_list = list()

    for test_id, test_name in enumerate(pre_process.name_list):
        date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        date_id_saving_list.append(date_id)
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
            f"date_id_{date_id}",
        ]
        # _splited_test_name = test_name.split("_")

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
            "has_nrem2_bias": HAS_NREM2_BIAS,
            "model_type": ENN_TAG,
        }
        main(
            name=f"code_{ENN_TAG}",
            project=WANDB_PROJECT,
            pre_process=pre_process,
            train=train,
            test=test,
            epochs=EPOCHS,
            save_model=True,
            has_attention=HAS_ATTENTION,
            my_tags=my_tags,
            date_id=date_id,
            pse_data=PSE_DATA,
            test_name=test_name,
            has_inception=HAS_INCEPTION,
            batch_size=BATCH_SIZE,
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
    # json に書き込み
    JB.dump(
        keys=[
            ENN_TAG,
            DATA_TYPE,
            FIT_POS,
            f"stride_{str(STRIDE)}",
            f"kernel_{str(KERNEL_SIZE)}",
            "no_cleansing",
        ],
        value=date_id_saving_list,
    )
