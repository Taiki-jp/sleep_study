import sys
import tensorflow as tf
import os
import datetime
import wandb
from wandb.keras import WandbCallback
from pre_process.pre_process import PreProcess
from nn.model_base import EDLModelBase, edl_classifier_1d
from nn.losses import EDLLoss

tf.random.set_seed(100)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main(
    name,
    project,
    train,
    test,
    pre_process,
    epochs=1,
    save_model=False,
    my_tags=None,
    batch_size=32,
    n_class=5,
    pse_data=False,
    test_name=None,
    date_id=None,
    has_attention=False,
    has_inception=True,
    utils=None,
    data_type=None,
    sample_size=0,
    kernel_size=0,
    stride=0,
    fit_pos="",
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

    # wandbの初期化
    wandb.init(
        name=name,
        project=project,
        tags=my_tags,
        config={
            "test name": test_name,
            "date id": date_id,
            "batch_size": batch_size,
            "attention": has_attention,
            "inception": has_inception,
            "n_class": n_class,
            "sample_size": sample_size,
            "epochs": epochs,
            "kernel": kernel_size,
            "stride": stride,
            "fit_pos": fit_pos,
        },
        sync_tensorboard=True,
        dir=pre_process.my_env.project_dir,
    )

    # モデルの作成とコンパイル
    if data_type == "spectrum":
        shape = (int(KERNEL_SIZE / 2), 1)
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
    )
    model = EDLModelBase(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=EDLLoss(K=n_class, annealing=0.1),
        metrics=["accuracy"],
    )

    # tensorboard作成
    log_dir = os.path.join(
        pre_process.my_env.project_dir, "my_edl", test_name, date_id
    )
    tf_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        epochs=epochs,
        callbacks=[tf_callback, WandbCallback()],
        verbose=2,
    )

    if save_model:
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

    # ハイパーパラメータの設定
    TEST_RUN = False
    HAS_ATTENTION = True
    PSE_DATA = False
    HAS_INCEPTION = True
    IS_PREVIOUS = False
    IS_NORMAL = True
    EPOCHS = 100
    BATCH_SIZE = 32
    N_CLASS = 5
    KERNEL_SIZE = 1024
    STRIDE = 4
    SAMPLE_SIZE = 50000
    DATA_TYPE = "spectrum"
    FIT_POS = "middle"
    NORMAL_TAG = "normal" if IS_NORMAL else "sas"
    ATTENTION_TAG = "attention" if HAS_ATTENTION else "no-attention"
    PSE_DATA_TAG = "psedata" if PSE_DATA else "sleepdata"
    INCEPTION_TAG = "inception" if HAS_INCEPTION else "no-inception"
    WANDB_PROJECT = "test" if TEST_RUN else "master"

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
        load_all=True,
        pse_data=PSE_DATA,
    )

    for test_id, test_name in enumerate(pre_process.name_list):
        date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        (train, test) = pre_process.split_train_test_from_records(
            datasets, test_id=test_id, pse_data=PSE_DATA
        )
        # tagの設定
        my_tags = [
            f"{test_name}",
            PSE_DATA_TAG,
            ATTENTION_TAG,
            INCEPTION_TAG,
            DATA_TYPE,
            FIT_POS,
            f"kernel_{KERNEL_SIZE}",
            f"stride_{STRIDE}",
            f"sample_{SAMPLE_SIZE}",
        ]

        main(
            name=f"edl-{test_name}",
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
            fit_pos=FIT_POS,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
        )

        # testの時は一人の被験者で止める
        if TEST_RUN:
            break
