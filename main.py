import tensorflow as tf
import os
import datetime
import wandb
from data_analysis.utils import Utils
from wandb.keras import WandbCallback
from pre_process.pre_process import PreProcess
from pre_process.load_sleep_data import LoadSleepData
from nn.model_base import EDLModelBase, edl_classifier_1d
from nn.losses import EDLLoss

tf.random.set_seed(100)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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
):

    # データセットの作成
    (x_train, y_train), (x_test, y_test) = pre_process.make_dataset(
        train=train,
        test=test,
        is_storchastic=False,
        pse_data=pse_data,
        to_one_hot_vector=False,
        each_data_size=5000,
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
        },
        sync_tensorboard=True,
        dir=utils.project_dir,
    )

    # モデルの作成とコンパイル
    if data_type == "spectrum":
        shape = (512, 1)
    elif data_type == "spectrogram":
        shape = (128, 512, 1)
    else:
        # correct here based on your model
        shape = (512, 1)
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
    log_dir = os.path.join(utils.project_dir, "my_edl", test_name, date_id)
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
        path = os.path.join(utils.models_dir, test_name, date_id)
        model.save(path)
    wandb.finish()


if __name__ == "__main__":
    # 環境設定
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
    ATTENTION_TAG = "attention" if HAS_ATTENTION else "no-attention"
    PSE_DATA = False
    PSE_DATA_TAG = "psedata" if PSE_DATA else "sleepdata"
    EPOCHS = 100
    HAS_INCEPTION = True
    INCEPTION_TAG = "inception" if HAS_INCEPTION else "no-inception"
    WANDB_PROJECT = "test" if PSE_DATA else "edl-test"
    BATCH_SIZE = 32
    N_CLASS = 5
    DATA_TYPE = "spectrum"

    # オブジェクトの作成
    load_sleep_data = LoadSleepData(data_type=DATA_TYPE, n_class=N_CLASS)
    pre_process = PreProcess(load_sleep_data)
    datasets = load_sleep_data.load_data(
        load_all=True, pse_data=PSE_DATA, fit_pos="middle"
    )
    utils = Utils(file_reader=load_sleep_data.fr)

    for test_id, test_name in enumerate(load_sleep_data.sl.added_name_list):
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
            utils=utils,
            data_type=DATA_TYPE,
        )

        # testの時は一人の被験者で止める
        if TEST_RUN:
            break
