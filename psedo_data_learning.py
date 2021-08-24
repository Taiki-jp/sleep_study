import random
import tensorflow as tf
import numpy as np
import os
import sys
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from tensorflow.python.ops.gen_batch_ops import batch
from tensorflow.python.ops.numpy_ops.np_arrays import ndarray
from nn.losses import EDLLoss
import datetime
from data_analysis.py_color import PyColor
from data_analysis.utils import Utils
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


# @tf.function
def decorder(x: Tensor, hidden_unit: int, output_unit: int) -> Tensor:
    x = tf.keras.layers.Dense(hidden_unit)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(output_unit)(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def my_argmax(array: np.ndarray, axis: int, n_classes: int) -> np.ndarray:
    array_max = np.argmax(array, axis=axis)
    array_min = np.argmin(array, axis=axis)
    fixed_array = []
    # 最大値と最小値が一致する場合は1. random に値を返す
    for _max, _min in zip(array_max, array_min):
        if _max == _min:
            fixed_array.append(random.randint(0, n_classes - 1))
        else:
            fixed_array.append(_max)
    return np.array(fixed_array)


def main(
    x_train: Tensor,
    y_train: Tensor,
    utils: Utils,
    val_dataset: tuple,
    date_id: datetime.datetime,
    epochs: int = 50,
    input_unit: int = 2,
    hidden_unit: int = 3,
    n_class: int = 2,
    annealing_param: float = 1.0,
    subnet_starting_point: float = 0.5,
    project_name: str = "h-enn",
    experiment_type: str = "",
    sample_num: int = 100,
    batch_size: int = 16,
    unc_threthold: float = 0.5,
):
    # モデルの作成
    output_unit = n_class
    shape = tf.keras.Input(input_unit)
    outputs = decorder(
        x=shape, output_unit=output_unit, hidden_unit=hidden_unit
    )
    model = tf.keras.Model(inputs=shape, outputs=outputs)
    # モデルの構成
    print(PyColor.RED_FLASH, model.summary(), PyColor.END)
    # モデルのコンパイル
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=EDLLoss(K=n_class, annealing=annealing_param),
        metrics=["accuracy"],
    )
    # モデルの学習
    model.fit(
        x_train, y_train, batch_size=batch_size, epochs=epochs * 10, verbose=2
    )

    # 不確かさの高いデータを分類
    def _separate_unc_data(wants_histgram=True) -> tuple:
        evidence = model(x_train, training=False)
        alpha = evidence + 1
        unc = n_class / tf.reduce_sum(alpha, axis=1, keepdims=True)
        y_pred = alpha / tf.reduce_sum(alpha, axis=1, keepdims=True)
        y_pred = my_argmax(y_pred, axis=1, n_classes=n_class)
        mask = unc < unc_threthold
        if wants_histgram:
            utils.u_hist2Wandb(
                y=y_train,
                evidence=evidence,
                alpha=alpha,
                unc=unc,
                y_pred=y_pred,
                train_or_test="train",
                test_label=TEST_NAME,
                date_id=date_id,
                separate_each_ss=False,
            )

        return (
            tf.boolean_mask(x_train, mask.numpy().reshape(200)),
            tf.boolean_mask(y_train, mask.numpy().reshape(200)),
        )

    # モデルを新しく作成して新しいデータで学習
    (_x_train, _y_train) = _separate_unc_data()
    print(_y_train.shape)
    # 全てのデータが削られなかったら新しいモデルでも学習
    if _x_train.shape[0] != 0:
        _model = tf.keras.Model(inputs=shape, outputs=outputs)
        _model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=EDLLoss(K=n_class, annealing=annealing_param),
            metrics=["accuracy"],
        )
        _model.fit(
            _x_train, _y_train, batch_size=batch_size, epochs=epochs, verbose=2
        )
    else:
        print(PyColor.RED_FLASH, "すべてのデータが分離されました", PyColor.END)

if __name__ == "__main__":
    import datetime
    import wandb

    # 環境設定
    CALC_DEVICE = "cpu"
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

    # ANCHOR: ハイパラの設定
    TEST_RUN = False
    TEST_NAME = "test"
    DATA_TYPE = "type01"
    # code 名によって実験を分類
    # code_C(ompare)S(eed)
    RUN_NAME = "code_CS"
    EXPERIMENT_TYPE = "selective"
    HIDDEN_DIM = 8
    EPOCHS = 100
    N_CLASS = 2
    PROJECT_NAME = "test000"
    SAMPLE_NUM = 100
    ANNEALING_RATIO = 0.5
    # TODO: 誤差関数の重みづけの活性化関数と対応付ける
    utils = Utils()
    # データを生成
    (x_train, x_test), (y_train, y_test) = utils.make_2d_psedo_data(
        row=SAMPLE_NUM, col=2, data_type=DATA_TYPE
    )

    # seed でループを回す
    for fixed_seed in range(100):
        tf.random.set_seed(fixed_seed)
        # y_train の変換
        y_train = tf.cast(np.array(y_train), dtype=tf.float32)
        date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tagの設定
        wandb_tags = [
            TEST_NAME,
            DATA_TYPE,
            EXPERIMENT_TYPE,
            f"H:{str(HIDDEN_DIM)}",
            f"E:{str(EPOCHS)}",
            f"N:{str(N_CLASS)}",
            f"seed:{str(fixed_seed)}",
            f"AR:{str(ANNEALING_RATIO)}",
        ]

        wandb_config = {
            "test name": TEST_NAME,
            "data type": DATA_TYPE,
            "experiment type": EXPERIMENT_TYPE,
            "hidden_dim": HIDDEN_DIM,
            "epochs": EPOCHS,
            "n class": N_CLASS,
            "seed": fixed_seed,
        }

        wandb_saved_dir = os.path.join(os.environ["sleep"])
        # wandbの初期化
        wandb.init(
            name=RUN_NAME,
            project=PROJECT_NAME,
            tags=wandb_tags,
            config=wandb_config,
            dir=wandb_saved_dir,
        )
        # 何週目かの表示
        print(PyColor.GREEN_FLASH, f"{fixed_seed} SEED", PyColor.END)
        main(
            x_train=x_train,
            y_train=y_train,
            date_id=date_id,
            val_dataset=None,
            epochs=EPOCHS,
            input_unit=2,
            hidden_unit=HIDDEN_DIM,
            n_class=N_CLASS,
            project_name=PROJECT_NAME,
            experiment_type=EXPERIMENT_TYPE,
            utils=utils,
            annealing_param=ANNEALING_RATIO
        )
        # git の作成
        root_dir = os.path.join(os.environ["sleep"], "figures")
        each_dir_name_list = [
            "first_model",
            "second_model",
            "merged_model",
        ]
        saved_path_list = [
            os.path.join(
                root_dir,
                each_dir_name_list[i],
                "check_uncertainty",
                date_id,
            )
            for i in range(3)
        ]
        # for saved_path in saved_path_list:
        #     utils.make_gif(saved_path=saved_path)

        wandb.finish()

        if TEST_RUN:
            print(PyColor.RED_FLASH, "*** finish test run ***", PyColor.END)   
            break
