import random
import tensorflow as tf
import numpy as np
import os
import sys
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from nn.losses import EDLLoss
import datetime
from data_analysis.py_color import PyColor
from data_analysis.utils import Utils
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


# import matplotlib.pyplot as plt
# from nn.model_base import edl_classifier4psedo_data, EDLModelBase

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# @tf.function
def decorder(x: Tensor, hidden_unit: KerasTensor) -> Tensor:
    x = tf.keras.layers.Dense(hidden_unit)(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


# @tf.function
def classifier(x: Tensor, output_unit: int) -> Tensor:
    x = tf.keras.layers.Dense(output_unit)(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


# @tf.function
def nested_classifier(
    x: Tensor, converted_space_dim: int, output_unit: int
) -> Tensor:
    x = tf.keras.layers.Dense(converted_space_dim)(x)
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
    train_dataset: tuple,
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
):
    # output dim
    output_unit = n_class
    shape = (input_unit,)
    inputs = tf.keras.Input(shape=shape)
    # hidden_shape is (batch, 192)
    shape = (hidden_unit,)
    hidden_inputs = tf.keras.Input(shape=shape)
    # NOTE: outputの方は次元数を入れるだけ
    hidden = decorder(x=inputs, hidden_unit=hidden_unit)
    output_main = classifier(x=hidden_inputs, output_unit=output_unit)
    output_nested = nested_classifier(
        x=hidden_inputs,
        converted_space_dim=hidden_unit,
        output_unit=output_unit,
    )
    # decord model
    decord_nn = tf.keras.Model(inputs=inputs, outputs=hidden)
    # main classifier
    classifier_main_model = tf.keras.Model(
        inputs=hidden_inputs, outputs=output_main
    )
    # nested classifier
    classifier_sub_model = tf.keras.Model(
        inputs=hidden_inputs, outputs=output_nested
    )
    # 最適化関数の設定
    optimizer = tf.keras.optimizers.Adam()
    # 損失関数
    loss_class = EDLLoss(K=n_class, annealing=0)
    # モデルの構成
    # print(PyColor.RED_FLASH, decord_nn.summary(), PyColor.END)
    # print(PyColor.YELLOW, classifier_main_model.summary(), PyColor.END)
    # print(PyColor.GREEN_FLASH, classifier_sub_model.summary(), PyColor.END)

    # メトリクスの作成
    # true side : カテゴリカルな状態，pred side : クラスの次元数（ソフトマックスをかける前）
    # val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    epoch_loss_main_avg = tf.keras.metrics.Mean()
    epoch_loss_sub_avg = tf.keras.metrics.Mean()
    # サマリーライターのセットアップ
    current_time = date_id
    train_log_dir = os.path.join(
        os.environ["sleep"], "logs", "gradient_tape", current_time, "train"
    )
    if not os.path.exists:
        print(PyColor.RED_FLASH, f"make {train_log_dir}", PyColor.END)
        os.makedirs(train_log_dir)
    # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # test_log_dir = os.path.join(
    #     os.environ["sleep"], "logs", "gradient_tape", current_time, "test"
    # )
    # test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    def _make_image(
        x_train: Tensor,
        y_train: Tensor,
        iter: int,
    ):
        # 結果の出力(訓練データそのまま)
        # TODO: mainネットワークの予測
        hidden = decord_nn(x_train, training=False)
        evidence_main = classifier_main_model(hidden, training=False)
        alpha_main = evidence_main + 1
        y_pred_main = alpha_main / tf.reduce_sum(
            alpha_main, axis=1, keepdims=True
        )
        unc_main = n_class / tf.reduce_sum(alpha_main, axis=1, keepdims=True)
        # TODO: subネットワークの予測
        evidence_sub = classifier_sub_model(hidden, training=False)
        alpha_sub = evidence_sub + 1
        y_pred_sub = alpha_sub / tf.reduce_sum(
            alpha_sub, axis=1, keepdims=True
        )
        unc_sub = n_class / tf.reduce_sum(alpha_sub, axis=1, keepdims=True)
        # TODO: mergedネットワークの予測
        y_pred_merged = y_pred_main * (1 - unc_main) + y_pred_sub * unc_main

        def __draw(
            id: str,
            evidence: Tensor,
            alpha: Tensor,
            y_pred: Tensor,
            unc: Tensor,
            n_class: int,
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
            _y_train = tf.one_hot(y_train, depth=n_class)
            loss = loss_fn.call(_y_train, evidence)
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

        # メインネットワークの図
        __draw(
            id="main_network",
            evidence=evidence_main,
            alpha=alpha_main,
            y_pred=y_pred_main,
            unc=unc_main,
            n_class=n_class,
        )
        # サブネットワークの図
        __draw(
            id="sub_network",
            evidence=evidence_sub,
            alpha=alpha_sub,
            y_pred=y_pred_sub,
            unc=unc_sub,
            n_class=n_class,
        )
        # マージの図（予測以外いらない）
        __draw(
            id="merged_network",
            evidence=evidence_sub,
            alpha=alpha_sub,
            y_pred=y_pred_merged,
            unc=unc_sub,
            n_class=n_class,
        )

    # エポックのループ
    for epoch in range(epochs):
        # TODO: epoch に合わせてアニーリングを調整する
        loss_class.annealing = min(1, annealing_param * (epoch / epochs))
        print(f"エポック:{epoch + 1}")
        # エポック内のバッチサイズごとのループ
        for _, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            # 勾配を計算
            with tf.GradientTape(persistent=True) as tape_main:
                hidden_main = decord_nn(x_batch_train, training=True)
                evidence_main = classifier_main_model(
                    hidden_main, training=True
                )
                alpha_main = evidence_main + 1
                y_pred_train = alpha_main / tf.reduce_sum(
                    alpha_main, axis=1, keepdims=True
                )
                unc_main = n_class / tf.reduce_sum(
                    alpha_main, axis=1, keepdims=True
                )
                loss_value_main = loss_class.call(
                    tf.keras.utils.to_categorical(
                        y_batch_train, num_classes=n_class
                    ),
                    evidence_main,
                )
                # epoch が指定回数以上経過したのちにサブネットワークの順伝搬
                if (
                    epoch / epochs > subnet_starting_point
                    and experiment_type == "has_subnet"
                ):
                    evidence_sub = classifier_sub_model(
                        hidden_main, training=True
                    )
                    alpha_sub = evidence_sub + 1
                    y_pred_sub = alpha_sub / tf.reduce_sum(
                        alpha_sub, axis=1, keepdims=True
                    )
                    loss_value_sub = loss_class.call(
                        tf.keras.utils.to_categorical(
                            y_batch_train, num_classes=n_class
                        ),
                        evidence_sub,
                        unc_main,
                    )
                    # 進捗の記録
                    epoch_loss_sub_avg(loss_value_sub)
            # epoch が指定回数以上経過したのちにサブネットワークの逆伝搬
            if (
                epoch / epochs > subnet_starting_point
                and experiment_type == "has_subnet"
            ):
                grads_sub = tape_main.gradient(
                    loss_value_sub, classifier_sub_model.trainable_weights
                )
                optimizer.apply_gradients(
                    zip(grads_sub, classifier_sub_model.trainable_weights)
                )
                y_merged = (
                    1 - unc_main
                ) * y_pred_train + unc_main * y_pred_sub
                train_acc_metric.update_state(y_batch_train, y_merged)
            else:
                train_acc_metric.update_state(y_batch_train, y_pred_train)

            [grads_main, grads_exploit] = tape_main.gradient(
                loss_value_main,
                [
                    classifier_main_model.trainable_weights,
                    decord_nn.trainable_weights,
                ],
            )
            optimizer.apply_gradients(
                zip(grads_main, classifier_main_model.trainable_weights)
            )
            optimizer.apply_gradients(
                zip(grads_exploit, decord_nn.trainable_weights)
            )
            # 進捗の記録
            epoch_loss_main_avg(loss_value_main)

        # 画像の作成
        _make_image(
            x_train=x_train,
            y_train=y_train,
            iter=epoch,
        )
        # サマリーライターへの書き込み
        # with train_summary_writer.as_default():
        #     tf.summary.scalar("loss", epoch_loss_main_avg.result(), step=epoch)
        #     tf.summary.scalar(
        #         "accuracy", train_acc_metric.result(), step=epoch
        #     )
        # ヒストグラムのログ
        # デコーダー
        # tf.summary.histogram(
        #     "decorder/weights",
        #     decord_nn.get_layer("dense").weights[0],
        #     step=epoch,
        # )
        # tf.summary.histogram(
        #     "decorder/bias",
        #     decord_nn.get_layer("dense").weights[1],
        #     step=epoch,
        # )
        # # メインの分類器
        # tf.summary.histogram(
        #     "classifier_main/weights",
        #     classifier_main_model.get_layer("dense_1").weights[0],
        #     step=epoch,
        # )
        # tf.summary.histogram(
        #     "classifier_main/bias",
        #     classifier_main_model.get_layer("dense_1").weights[1],
        #     step=epoch,
        # )
        # # サブの分類器
        # tf.summary.histogram(
        #     "classifier_sub/weights01",
        #     classifier_sub_model.get_layer("dense_2").weights[0],
        #     step=epoch,
        # )
        # tf.summary.histogram(
        #     "classifier_sub/bias01",
        #     classifier_sub_model.get_layer("dense_2").weights[1],
        #     step=epoch,
        # )
        # tf.summary.histogram(
        #     "classifier_sub/weights02",
        #     classifier_sub_model.get_layer("dense_3").weights[0],
        #     step=epoch,
        # )
        # tf.summary.histogram(
        #     "classifier_sub/bias02",
        #     classifier_sub_model.get_layer("dense_3").weights[1],
        #     step=epoch,
        # )
        # エポックの終わりにメトリクスを表示する
        print(
            f"訓練一致率：{train_acc_metric.result():.2%}",
            f"訓練損失(main):{epoch_loss_main_avg.result():.5f}",
            f"訓練損失(merged):{epoch_loss_sub_avg.result():.5f}",
        )
        # wandbにログを送る（TODO：pre, rec, f-mも送る)
        wandb.log(
            {
                "train_acc": train_acc_metric.result(),
                "train_loss(main)": epoch_loss_main_avg.result(),
                "train_loss(merged)": epoch_loss_sub_avg.result(),
            }
        )
        # エポックの終わりに訓練メトリクスを初期化
        train_acc_metric.reset_states()


if __name__ == "__main__":
    import datetime
    import wandb

    # ANCHOR: ハイパラの設定
    TEST_NAME = "test"
    DATA_TYPE = "type03"
    # code 名によって実験を分類
    # code_C(ompare)S(eed)
    RUN_NAME = "code_CS"
    EXPERIMENT_TYPE = ["has_subnet", "no_subnet"]
    HIDDEN_DIM = 8
    EPOCHS = 50
    N_CLASS = 2
    ANNEALING_RATIO = 16
    SUBNET_STARTING_POINNT = 0.5
    PROJECT_NAME = "h-enn"
    SAMPLE_NUM = 100
    # TODO: 誤差関数の重みづけの活性化関数と対応付ける
    WEITED_ACTIVATION = "none"
    utils = Utils()

    # seed でループを回す
    for fixed_seed in range(100):
        for experiment_type in EXPERIMENT_TYPE:

            tf.random.set_seed(fixed_seed)

            (x_train, x_test), (y_train, y_test) = utils.archimedes_spiral(
                row=SAMPLE_NUM, col=2, x_bias=0, y_bias=0
            )
            # カスタムトレーニングのために作成
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (x_train, y_train)
            )
            train_dataset = train_dataset.shuffle(
                buffer_size=x_train.shape[0]
            ).batch(8)
            date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # tagの設定
            wandb_tags = [
                TEST_NAME,
                DATA_TYPE,
                experiment_type,
                f"H:{str(HIDDEN_DIM)}",
                f"E:{str(EPOCHS)}",
                f"N:{str(N_CLASS)}",
                f"AR:{str(ANNEALING_RATIO)}",
                f"SSP:{str(SUBNET_STARTING_POINNT)}",
                f"seed:{str(fixed_seed)}",
                f"av:{WEITED_ACTIVATION}",
            ]

            wandb_config = {
                "test name": TEST_NAME,
                "data type": DATA_TYPE,
                "experiment type": experiment_type,
                "hidden_dim": HIDDEN_DIM,
                "epochs": EPOCHS,
                "n class": N_CLASS,
                "annealing ratio": ANNEALING_RATIO,
                "subnet_starting_point": SUBNET_STARTING_POINNT,
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
                train_dataset=train_dataset,
                val_dataset=None,
                epochs=EPOCHS,
                input_unit=2,
                hidden_unit=HIDDEN_DIM,
                n_class=N_CLASS,
                annealing_param=ANNEALING_RATIO,
                subnet_starting_point=SUBNET_STARTING_POINNT,
                project_name=PROJECT_NAME,
                experiment_type=experiment_type,
                utils=utils,
            )
            # git の作成
            root_dir = os.path.join(os.environ["sleep"], "figures")
            each_dir_name_list = [
                "main_network",
                "sub_network",
                "merged_network",
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
            for saved_path in saved_path_list:
                utils.make_gif(saved_path=saved_path)

            wandb.finish()
