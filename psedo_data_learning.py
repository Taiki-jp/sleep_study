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
        loss=EDLLoss(K=n_class, annealing=0.1),
        metrics=["accuracy"],
    )
    # モデルの学習
    model.fit(
        x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2
    )

    # 不確かさの高いデータを分類
    def _separate_unc_data() -> tuple:
        evidence = model(x_train, training=False)
        alpha = evidence + 1
        unc = n_class / tf.reduce_sum(alpha, axis=1, keepdims=True)
        mask = unc < unc_threthold

        return (
            tf.boolean_mask(x_train, mask.numpy().reshape(200)),
            tf.boolean_mask(y_train, mask.numpy().reshape(200)),
        )

    # モデルを新しく作成して新しいデータで学習
    (_x_train, _y_train) = _separate_unc_data()
    _model = tf.keras.Model(inputs=shape, outputs=outputs)
    _model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=EDLLoss(K=n_class, annealing=0.1),
        metrics=["accuracy"],
    )
    _model.fit(
        _x_train, _y_train, batch_size=batch_size, epochs=epochs, verbose=2
    )

    def _make_image(
        x_train: Tensor,
        y_train: Tensor,
        iter: int,
    ):
        # 結果の出力(訓練データそのまま)
        # TODO: mainネットワークの予測
        evidence = model(x_train, training=False)
        alpha = evidence + 1
        y_pred = alpha / tf.reduce_sum(alpha, axis=1, keepdims=True)
        unc = n_class / tf.reduce_sum(alpha, axis=1, keepdims=True)
        # TODO: subネットワークの予測
        _evidence = _model(_x_train, training=False)
        _alpha = _evidence + 1
        _y_pred = _alpha / tf.reduce_sum(_alpha, axis=1, keepdims=True)
        _unc = n_class / tf.reduce_sum(_alpha, axis=1, keepdims=True)

        def __draw(
            id: str,
            evidence: Tensor,
            y_pred: Tensor,
            unc: Tensor,
            n_class: int,
            x_train: Tensor,
            y_train: Tensor,
        ):
            figure = plt.figure(figsize=(12, 4))
            # 1. 正解の散布図
            # TODO: クラスが順番に並んでない時にも対応できるように変更
            ax = figure.add_subplot(1, 4, 1)
            for x, label in zip(x_train, y_train):
                if label == 0:
                    ax.scatter(x[0], x[1], c="r")
                elif label == 1:
                    ax.scatter(x[0], x[1], c="b")
                else:
                    print("exception has occured")
                    sys.exit(1)
            ax.set_title("true")
            # 2. 予測の散布図
            ax = figure.add_subplot(142)
            ax.set_title("pred")
            # 予測の確率ベクトルが等しいときはランダムに値を返す
            y_pred_ctg = my_argmax(array=y_pred, axis=1, n_classes=n_class)

            for x, label in zip(x_train, y_pred_ctg):
                if label == 0:
                    ax.scatter(x[0], x[1], c="r")
                elif label == 1:
                    ax.scatter(x[0], x[1], c="b")
                else:
                    print("exception has occured")
                    sys.exit(1)
            # 3. 不確かさの分布
            ax = figure.add_subplot(143)
            ax.set_title("unc")
            im = ax.scatter(
                x_train[:, 0],
                x_train[:, 1],
                c=unc,
                cmap="Blues",
                vmin=0,
                vmax=1,
            )
            figure.colorbar(im)

            # 4. ロスの分布
            loss_fn = EDLLoss(K=n_class, annealing=1)
            __y_train = tf.one_hot(y_train, depth=n_class)
            loss = loss_fn.call(__y_train, evidence)
            ax = figure.add_subplot(144)
            ax.set_title("loss")
            im = ax.scatter(
                x_train[:, 0],
                x_train[:, 1],
                c=loss,
                cmap="Blues",
                vmin=0,
                vmax=1,
            )
            figure.colorbar(im)

            # plt.legend()
            save_path = os.path.join(
                os.environ["sleep"],
                "figures",
                id,
                "check_uncertainty",
                date_id,
            )
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if iter < 10:
                plt.savefig(os.path.join(save_path, f"00{iter}.png"))
            elif iter < 100:
                plt.savefig(os.path.join(save_path, f"0{iter}.png"))
            elif iter < 1000:
                plt.savefig(os.path.join(save_path, f"{iter}.png"))
            plt.cla()
            plt.clf()
            plt.close()

        # メインモデルの図
        __draw(
            id="first_model",
            evidence=evidence,
            alpha=alpha,
            y_pred=y_pred,
            unc=unc,
            n_class=n_class,
            x_train=x_train,
            y_train=y_train,
        )
        # サブモデルの図
        __draw(
            id="second_model",
            evidence=_evidence,
            alpha=_alpha,
            y_pred=_y_pred,
            unc=_unc,
            n_class=n_class,
            x_train=_x_train,
            y_train=_y_train,
        )


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
    TEST_NAME = "test"
    DATA_TYPE = "type03"
    # code 名によって実験を分類
    # code_C(ompare)S(eed)
    RUN_NAME = "code_CS"
    EXPERIMENT_TYPE = ["selective"]
    HIDDEN_DIM = 8
    EPOCHS = 50
    N_CLASS = 2
    PROJECT_NAME = "test"
    SAMPLE_NUM = 100
    # TODO: 誤差関数の重みづけの活性化関数と対応付ける
    utils = Utils()

    # seed でループを回す
    for fixed_seed in range(100):
        for experiment_type in EXPERIMENT_TYPE:

            tf.random.set_seed(fixed_seed)

            (x_train, x_test), (y_train, y_test) = utils.point_symmetry_data(
                row=SAMPLE_NUM, col=2, x_bias=0, y_bias=0
            )
            # y_train の変換
            y_train = tf.cast(np.array(y_train), dtype=tf.float32)
            date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # tagの設定
            wandb_tags = [
                TEST_NAME,
                DATA_TYPE,
                experiment_type,
                f"H:{str(HIDDEN_DIM)}",
                f"E:{str(EPOCHS)}",
                f"N:{str(N_CLASS)}",
                f"seed:{str(fixed_seed)}",
            ]

            wandb_config = {
                "test name": TEST_NAME,
                "data type": DATA_TYPE,
                "experiment type": experiment_type,
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
                experiment_type=experiment_type,
                utils=utils,
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
