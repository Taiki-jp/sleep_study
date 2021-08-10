import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from nn.losses import EDLLoss
import datetime
from data_analysis.py_color import PyColor

# import matplotlib.pyplot as plt
# from nn.model_base import edl_classifier4psedo_data, EDLModelBase

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.random.set_seed(0)


# 仮データの作成
def psedo_data(row: int, col: int, x_bias: int, y_bias: int) -> tuple:
    # 極座標で考える
    row, _ = (100, 2)
    r_class0 = tf.random.uniform(shape=(row,), minval=0, maxval=0.6)
    theta_class0 = tf.random.uniform(shape=(row,), minval=0, maxval=np.pi * 2)
    r_class1 = tf.random.uniform(shape=(row,), minval=0.5, maxval=1)
    theta_class1 = tf.random.uniform(shape=(row,), minval=0, maxval=np.pi * 2)
    input_class0 = (
        x_bias + r_class0 * np.cos(theta_class0),
        y_bias + r_class0 * np.sin(theta_class0),
    )
    input_class1 = (
        x_bias + r_class1 * np.cos(theta_class1),
        y_bias + r_class1 * np.sin(theta_class1),
    )
    x_train = tf.concat([input_class0, input_class1], axis=1)
    x_train = tf.transpose(x_train)
    y_train_0 = [0 for _ in range(row)]
    y_train_1 = [1 for _ in range(row)]
    y_train = y_train_0 + y_train_1
    x_test = None
    y_test = None
    return (x_train, x_test), (y_train, y_test)


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


def main(
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
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # test_log_dir = os.path.join(
    #     os.environ["sleep"], "logs", "gradient_tape", current_time, "test"
    # )
    # test_summary_writer = tf.summary.create_file_writer(test_log_dir)

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
                    alpha_main,
                )
                # epoch が半分以上経過したのちにサブネットワークを学習開始
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
                        alpha_sub,
                        unc_main,
                    )
                    # 進捗の記録
                    epoch_loss_sub_avg(loss_value_sub)
            # ある回数以上になるとサブネットワークの学習を開始する
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

        # サマリーライターへの書き込み
        with train_summary_writer.as_default():
            tf.summary.scalar("loss", epoch_loss_main_avg.result(), step=epoch)
            tf.summary.scalar(
                "accuracy", train_acc_metric.result(), step=epoch
            )
            # ヒストグラムのログ
            # デコーダー
            tf.summary.histogram(
                "decorder/weights",
                decord_nn.get_layer("dense").weights[0],
                step=epoch,
            )
            tf.summary.histogram(
                "decorder/bias",
                decord_nn.get_layer("dense").weights[1],
                step=epoch,
            )
            # メインの分類器
            tf.summary.histogram(
                "classifier_main/weights",
                classifier_main_model.get_layer("dense_1").weights[0],
                step=epoch,
            )
            tf.summary.histogram(
                "classifier_main/bias",
                classifier_main_model.get_layer("dense_1").weights[1],
                step=epoch,
            )
            # サブの分類器
            tf.summary.histogram(
                "classifier_sub/weights01",
                classifier_sub_model.get_layer("dense_2").weights[0],
                step=epoch,
            )
            tf.summary.histogram(
                "classifier_sub/bias01",
                classifier_sub_model.get_layer("dense_2").weights[1],
                step=epoch,
            )
            tf.summary.histogram(
                "classifier_sub/weights02",
                classifier_sub_model.get_layer("dense_3").weights[0],
                step=epoch,
            )
            tf.summary.histogram(
                "classifier_sub/bias02",
                classifier_sub_model.get_layer("dense_3").weights[1],
                step=epoch,
            )
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
    DATA_TYPE = "type01"
    RUN_NAME = DATA_TYPE
    EXPERIMENT_TYPE = "has_subnet"
    # EXPERIMENT_TYPE = "no_subnet"
    HIDDEN_DIM = 8
    EPOCHS = 50
    N_CLASS = 2
    ANNEALING_RATIO = 16
    SUBNET_STARTING_POINNT = 0.5
    PROJECT_NAME = "h-enn"
    (x_train, x_test), (y_train, y_test) = psedo_data(
        row=100, col=2, x_bias=0, y_bias=0
    )
    # カスタムトレーニングのために作成
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=x_train.shape[0]).batch(
        8
    )
    date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tagの設定
    wandb_tags = [
        TEST_NAME,
        DATA_TYPE,
        EXPERIMENT_TYPE,
        str(HIDDEN_DIM),
        str(EPOCHS),
        str(N_CLASS),
        str(ANNEALING_RATIO),
        str(SUBNET_STARTING_POINNT),
    ]

    wandb_config = {
        "test name": TEST_NAME,
        "data type": DATA_TYPE,
        "experiment type": EXPERIMENT_TYPE,
        "hidden_dim": HIDDEN_DIM,
        "epochs": EPOCHS,
        "n class": N_CLASS,
        "annealing ratio": ANNEALING_RATIO,
        "subnet_starting_point": SUBNET_STARTING_POINNT,
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
    main(
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
        experiment_type=EXPERIMENT_TYPE,
    )
