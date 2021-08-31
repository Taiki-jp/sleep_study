import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # tensorflow を読み込む前のタイミングですると効果あり
import tensorflow as tf

tf.random.set_seed(0)
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.ops.array_ops import boolean_mask
from data_analysis.py_color import PyColor
import datetime
import wandb
from collections import Counter
from pre_process.pre_process import PreProcess
from nn.model_base import classifier4enn, spectrum_conv
from nn.losses import EDLLoss
from pre_process.json_base import JsonBase
import numpy as np
from data_analysis.utils import Utils


def main(
    name: str,
    utils: Utils,
    project: str,
    train: list,
    test: list,
    pre_process: PreProcess,
    annealing_param: float,
    model_type: str,
    epochs: int = 1,
    my_tags: list = None,
    batch_size: int = 32,
    n_class: int = 5,
    pse_data: bool = False,
    test_name: str = None,
    date_id: str = None,
    has_attention: bool = False,
    has_inception: bool = False,
    data_type: str = "",
    sample_size: int = 0,
    wandb_config: dict = dict(),
    kernel_size: int = 0,
    is_mul_layer: bool = False,
    has_dropout: bool = False,
    subnet_starting_point: float = 0,
    is_enn: bool = False,
    experiment_type: str = "",
    loaded_model_id: str = "",
    unc_threthold: float = 0,
):

    # データの選択
    def __selects_data() -> tuple:
        # データセットの作成(one-hot で処理を行う)
        # カテゴリカルに変換 TODO: make_dataset 内で onehot 表現に変えてもよいかチェック
        (x_train, y_train), (x_test, y_test) = pre_process.make_dataset(
            train=train,
            test=test,
            is_storchastic=False,
            pse_data=pse_data,
            each_data_size=sample_size,
            to_one_hot_vector=False,
        )

        # データセットの数（変換前）
        ss_train_dict_before = Counter(y_train)
        ss_test_dict_before = Counter(y_test)

        if experiment_type == "no_cleansing":
            pass

        elif (
            experiment_type == "positive_cleansing"
            or experiment_type == "negative_cleansing"
        ):
            print(PyColor.GREEN, "データクレンジングを行います", PyColor.END)
            __model_path = os.path.join(
                pre_process.my_env.models_dir, test_name, loaded_model_id
            )
            __loaded_model = tf.keras.models.load_model(
                __model_path,
                custom_objects={"EDLLoss": EDLLoss(K=n_class, annealing=0)},
            )
            # 訓練データのクレンジング
            __evidence_train = __loaded_model.predict(
                x_train, batch_size=batch_size
            )
            __alpha_train = __evidence_train + 1
            __unc_train = n_class / tf.reduce_sum(
                __alpha_train, axis=1, keepdims=True
            )
            __y_pred_train = utils.my_argmax(
                __alpha_train
                / tf.reduce_sum(__alpha_train, axis=1, keepdims=True)
            )
            # テストデータのクレンジング
            __evidence_test = __loaded_model.predict(
                x_train, batch_size=batch_size
            )
            __alpha_test = __evidence_test + 1
            __unc_test = n_class / tf.reduce_sum(
                __alpha_test, axis=1, keepdims=True
            )
            __y_pred_test = utils.my_argmax(
                __alpha_test
                / tf.reduce_sum(__alpha_test, axis=1, keepdims=True)
            )

            if experiment_type == "positive_cleansing":
                __mask_train = __unc_train < unc_threthold
                __mask_test = __unc_test < unc_threthold
            elif experiment_type == "negative_cleansing":
                __mask_train = __unc_train < unc_threthold
                __mask_test = __unc_test < unc_threthold
            else:
                print(
                    PyColor.RED_FLASH,
                    "データクレンジングの方法として正しいものを選択してください",
                    PyColor.END,
                )
                sys.exit(1)

            __mask_train = __mask_train.numpy().reshape(y_train.shape[0])
            __mask_test = __mask_test.numpy().reshape(y_test.shape[0])
            x_train = tf.boolean_mask(x_train, __mask_train)
            y_train = tf.boolean_mask(y_train, __mask_train)
            x_test = tf.boolean_mask(x_test, __mask_test)
            y_test = tf.boolean_mask(y_test, __mask_test)

        else:
            print(
                PyColor.RED_FLASH, "正しい experiment_type を指定してください", PyColor.END
            )
            sys.exit(1)

        # データセットの数（変換後）
        ss_train_dict_after = Counter(y_train)
        ss_test_dict_after = Counter(y_test)

        # カスタムトレーニングのために作成
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(
            buffer_size=x_train.shape[0]
        ).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        val_dataset = val_dataset.batch(batch_size)
        return (train_dataset, val_dataset), (
            ss_train_dict_before,
            ss_test_dict_before,
            ss_train_dict_after,
            ss_test_dict_after,
        )

    # データの投入
    (
        (train_dataset, val_dataset),
        (
            ss_train_dict_before,
            ss_test_dict_before,
            ss_train_dict_after,
            ss_test_dict_after,
        ),
    ) = __selects_data()

    # カスタムトレーニングのために作成

    # config の追加
    added_config = {
        "attention": has_attention,
        "inception": has_inception,
        "test wake before replaced": ss_test_dict_before[4],
        "test rem before replaced": ss_test_dict_before[3],
        "test nr1 before replaced": ss_test_dict_before[2],
        "test nr2 before replaced": ss_test_dict_before[1],
        "test nr34 before replaced": ss_test_dict_before[0],
        "train wake before replaced": ss_train_dict_before[4],
        "train rem before replaced": ss_train_dict_before[3],
        "train nr1 before replaced": ss_train_dict_before[2],
        "train nr2 before replaced": ss_train_dict_before[1],
        "train nr34 before replaced": ss_train_dict_before[0],
        "test wake after replaced": ss_test_dict_after[4],
        "test rem after replaced": ss_test_dict_after[3],
        "test nr1 after replaced": ss_test_dict_after[2],
        "test nr2 after replaced": ss_test_dict_after[1],
        "test nr34 after replaced": ss_test_dict_after[0],
        "train wake after replaced": ss_train_dict_after[4],
        "train rem after replaced": ss_train_dict_after[3],
        "train nr1 after replaced": ss_train_dict_after[2],
        "train nr2 after replaced": ss_train_dict_after[1],
        "train nr34 after replaced": ss_train_dict_after[0],
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

    # モデルの選択
    def __selects_model() -> tuple:
        # すべてのモデルで共通の設定
        inputs = tf.keras.Input(shape=shape)
        # 最適化関数の設定
        optimizer = tf.keras.optimizers.Adam()
        # モデルタイプがh-enn であれば隠れ層のサイズを指定してあげる必要がある
        if model_type == "h_enn":
            # hidden_shape is (batch, 192->288)
            hidden = spectrum_conv(
                x=inputs,
                has_attention=has_attention,
                has_inception=has_inception,
                is_mul_layer=is_mul_layer,
            )
            hidden_inputs = tf.keras.Input(shape=(288,))
            # NOTE: main の方は 特徴量空間の変換を持たない
            output_main = classifier4enn(
                x=hidden_inputs,
                has_dropout=has_dropout,
                hidden_dim=0,
                has_converted_space=False,
                n_class=n_class,
            )
            output_sub = classifier4enn(
                x=hidden_inputs,
                has_dropout=has_dropout,
                hidden_dim=288,
                has_converted_space=True,
                n_class=n_class,
            )

            exploit_model = tf.keras.Model(inputs=inputs, outputs=hidden)
            classifier_main_model = tf.keras.Model(
                inputs=hidden_inputs, outputs=output_main
            )
            classifier_sub_model = tf.keras.Model(
                inputs=hidden_inputs, outputs=output_sub
            )
            # 損失関数だけここで定義しておく
            loss_class = EDLLoss(K=n_class, annealing=0)
            return (
                exploit_model,
                classifier_main_model,
                classifier_sub_model,
            ), (loss_class, optimizer)

        elif model_type == "enn" or model_type == "dnn":
            outputs = spectrum_conv(
                x=inputs,
                has_attention=has_attention,
                has_inception=has_inception,
                is_mul_layer=is_mul_layer,
                is_out_logits=True,
                n_class=n_class,
            )
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            if model_type == "enn":
                loss_class = EDLLoss(K=n_class, annealing=0)
            else:
                # NOTE: dnn のみがキャッチされているのが確認できたら取り除いてよい
                try:
                    assert model_type == "dnn"
                except:
                    print(
                        PyColor.RED_FLASH,
                        "model_type のキャッチが総程度売りされていません",
                        PyColor.END,
                    )
                    sys.exti(1)
                loss_class = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True
                )
            return (model,), (loss_class, optimizer)

        else:
            print(PyColor.RED_FLASH, "知らないモデルタイプが指定されました", PyColor.END)
            sys.exit(1)

    # モデルの還元
    (model), (loss_class, optimizer) = __selects_model()

    # メトリクスの作成 NOTE: true side : カテゴリカルな状態，pred side : one-hot表現（ソフトマックスをかける前）
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    epoch_loss_main_avg = tf.keras.metrics.Mean()
    epoch_loss_sub_avg = tf.keras.metrics.Mean()

    def __train_epoch(
        epoch: int,
        dataset: DatasetV2,
        model_type: str,
    ):
        def __train_epoch_dnn():
            for _, (x_batch_train, y_batch_train) in enumerate(dataset):
                with tf.GradientTape(persistent=True) as tape:
                    y_pred = model[0](x_batch_train, training=True)
                    # NOTE: one-hot 表現でいい？
                    loss_value = loss_class.call(y_batch_train, y_pred)

                grads = tape.gradient(
                    loss_value,
                    model[0].trainable_weights,
                )
                optimizer.apply_gradients(
                    zip(grads, model[0].trainable_weights)
                )
                # one-hot 表現で送ること
                train_acc_metric.update_state(y_batch_train, y_pred)

        def __train_epoch_enn():
            loss_class.annealing = min(1, annealing_param * (epoch / epochs))
            for _, (x_batch_train, y_batch_train) in enumerate(dataset):
                with tf.GradientTape(persistent=True) as tape:
                    evidence = model[0](x_batch_train, training=True)
                    alpha = evidence + 1
                    unc = n_class / tf.reduce_sum(alpha, axis=1, keepdims=True)
                    y_pred = alpha / tf.reduce_sum(
                        alpha, axis=1, keepdims=True
                    )
                    loss_value = loss_class.call(
                        tf.keras.utils.to_categorical(y_batch_train, n_class),
                        evidence,
                    )

                grads = tape.gradient(
                    loss_value,
                    model[0].trainable_weights,
                )
                optimizer.apply_gradients(
                    zip(grads, model[0].trainable_weights)
                )
                # one-hot 表現で送ること
                train_acc_metric.update_state(y_batch_train, y_pred)

        def __train_epoch_h_enn():
            loss_class.annealing = min(1, annealing_param * (epoch / epochs))
            for _, (x_batch_train, y_batch_train) in enumerate(dataset):
                # エポック内のバッチサイズごとのループ
                # 勾配を計算
                with tf.GradientTape(persistent=True) as tape_main:
                    hidden_main = model[0](x_batch_train, training=True)
                    evidence_main = model[1](hidden_main, training=True)
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
                    if epoch / epochs > subnet_starting_point:
                        evidence_sub = model[2](hidden_main, training=True)
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
                if epoch / epochs > subnet_starting_point:
                    grads_sub = tape_main.gradient(
                        loss_value_sub, model[2].trainable_weights
                    )
                    optimizer.apply_gradients(
                        zip(grads_sub, model[2].trainable_weights)
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
                        model[1].trainable_weights,
                        model[0].trainable_weights,
                    ],
                )
                optimizer.apply_gradients(
                    zip(grads_main, model[1].trainable_weights)
                )
                optimizer.apply_gradients(
                    zip(grads_exploit, model[0].trainable_weights)
                )
                # 進捗の記録
                epoch_loss_main_avg(loss_value_main)

        if model_type == "dnn":
            return __train_epoch_dnn()
        elif model_type == "enn":
            return __train_epoch_enn()
        elif model_type == "h_enn":
            return __train_epoch_h_enn()
        else:
            print(PyColor.RED_FLASH, "知らないモデルタイプが来ました", PyColor.END)
            sys.exit(1)

    def __test(epoch: int, dataset: DatasetV2, model_type: str):
        def __test_dnn():
            for _, (x_batch_val, y_batch_val) in enumerate(dataset):
                val_logits = model[0](x_batch_val, training=False)
                val_acc_metric.update_state(y_batch_val, val_logits)

        def __test_enn():
            for x_batch_val, y_batch_val in val_dataset:
                evidence = model[0](x_batch_val, training=False)
                alpha = evidence + 1
                unc = n_class / tf.reduce_sum(alpha, axis=1, keepdims=True)
                y_pred = alpha / tf.reduce_sum(alpha, axis=1, keepdims=True)
                loss_value = loss_class.call(
                    tf.keras.utils.to_categorical(y_batch_val, n_class),
                    evidence,
                )
                val_acc_metric.update_state(y_batch_val, y_pred)

        def __test_h_enn():
            for x_batch_val, y_batch_val in val_dataset:
                hidden_main = model[0](x_batch_val, training=False)
                evidence_main = model[1](hidden_main, training=False)
                alpha_main = evidence_main + 1
                y_pred_main = alpha_main / tf.reduce_sum(
                    alpha_main, axis=1, keepdims=True
                )
                unc_main = n_class / tf.reduce_sum(
                    alpha_main, axis=1, keepdims=True
                )
                if epoch / epochs > subnet_starting_point:
                    evidence_sub = model[2](hidden_main, training=False)
                    alpha_sub = evidence_sub + 1
                    y_pred_sub = alpha_sub / tf.reduce_sum(
                        alpha_main, axis=1, keepdims=True
                    )
                    y_merged = (
                        1 - unc_main
                    ) * y_pred_main + unc_main * y_pred_sub
                    val_acc_metric.update_state(y_batch_val, y_merged)
                else:
                    val_acc_metric.update_state(y_batch_val, y_pred_main)
            # メトリクスを表示

        if model_type == "dnn":
            return __test_dnn()
        elif model_type == "enn":
            return __test_enn()
        elif model_type == "h_enn":
            return __test_h_enn()
        else:
            print(PyColor.RED_FLASH, "知らないモデルタイプが来ました", PyColor.END)
            sys.exit(1)

    # エポックのループ
    for epoch in range(epochs):
        print(PyColor.YELLOW, f"エポック:{epoch + 1}", PyColor.END)

        # 訓練
        __train_epoch(epoch, train_dataset, model_type)
        print(f"訓練一致率：{train_acc_metric.result():.2%}")

        # テスト
        __test(epoch, val_dataset, model_type)
        print(f"テスト一致率：{val_acc_metric.result():.2%}")

        # wandbにログを送る（：pre, rec, f-mも送る)
        log_info = {
            "train_acc": train_acc_metric.result(),
            "train_loss_main": epoch_loss_main_avg.result(),
            "train_loss_sub": epoch_loss_sub_avg.result(),
            "val_acc": val_acc_metric.result(),
        }
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()
        wandb.log(log_info)

    # モデルの保存（モデルの種類によって保存の方法を変更する）
    def __save_model():
        if model_type == "dnn" or model_type == "enn":
            model_path = os.path.join(
                pre_process.my_env.models_dir, test_name, date_id
            )
            model.save(model_path)
        elif model_type == "h_enn":
            path = os.path.join(pre_process.my_env.models_dir, test_name)
            exploit_model_path = os.path.join(path, "exploit")
            main_model_path = os.path.join(path, "main")
            sub_model_path = os.path.join(path, "sub")
            if not os.path.exists(exploit_model_path):
                os.makedirs(exploit_model_path)
            if not os.path.exists(main_model_path):
                os.makedirs(main_model_path)
            if not os.path.exists(sub_model_path):
                os.makedirs(sub_model_path)
            model[0].save(exploit_model_path + f"/{date_id}")
            model[1].save(main_model_path + f"/{date_id}")
            model[2].save(sub_model_path + f"/{date_id}")
        else:
            print(PyColor.RED_FLASH, "知らないモデルタイプが来ました", PyColor.END)
            sys.exit(1)

    # # wandbへの記録終了
    wandb.finish()


# 睡眠データに対するプログラム
# ANCHOR: main
if __name__ == "__main__":
    # 環境設定（基本的にいじるのここだけ）
    CALC_DEVICE = "gpu"
    DEVICE_ID = "0" if CALC_DEVICE == "gpu" else "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID
    if os.environ["CUDA_VISIBLE_DEVICES"] != "-1":
        tf.keras.backend.set_floatx("float32")
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # tf.config.run_functions_eagerly(True)  # これを使うと遅くなる
    else:
        print(PyColor.RED_FLASH, "cpuで計算します", PyColor.END)

    # ANCHOR: hyper_paramerter
    # ハイパーパラメータの設定
    TEST_RUN = True
    HAS_ATTENTION = True
    PSE_DATA = False
    HAS_INCEPTION = True
    IS_PREVIOUS = False
    IS_NORMAL = True
    IS_ENN = False
    HAS_NREM2_BIAS = True
    EPOCHS = 50
    BATCH_SIZE = 512
    N_CLASS = 5
    KERNEL_SIZE = 512
    STRIDE = 1024
    SAMPLE_SIZE = 5000
    ANNEALING_RATIO = 16
    SUBNET_STARTING_POINNT = 0.5
    EXPERIMENT_TYPES = (
        "no_cleansing",
        "positive_cleansing",
        "negative_cleansing",
    )
    EXPERIENT_TYPE = "positive_cleansing"  # ここで model
    MODEL_TYPE_LIST = [
        "dnn",
        "enn",
        "h_enn",
    ]
    MODEL_TYPE = "enn"
    DATA_TYPE = "spectrum"
    FIT_POS = "middle"
    NORMAL_TAG = "normal" if IS_NORMAL else "sas"
    ATTENTION_TAG = "attention" if HAS_ATTENTION else "no-attention"
    PSE_DATA_TAG = "psedata" if PSE_DATA else "sleepdata"
    INCEPTION_TAG = "inception" if HAS_INCEPTION else "no-inception"
    WANDB_PROJECT = "test" if TEST_RUN else "master000"
    ENN_TAG = "enn" if IS_ENN else "dnn"

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

    utils = Utils()
    # 記録用のjsonファイルを読み込む
    JB = JsonBase("../nn/model_id.json")
    JB.load()
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
            MODEL_TYPE,
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
            "model type": MODEL_TYPE,
            "experiment type": EXPERIENT_TYPE,
        }

        main(
            utils=utils,
            data_type=DATA_TYPE,
            name=test_name,
            project=WANDB_PROJECT,
            pre_process=pre_process,
            train=train,
            test=test,
            epochs=EPOCHS,
            has_attention=HAS_ATTENTION,
            my_tags=my_tags,
            date_id=date_id,
            pse_data=PSE_DATA,
            test_name=test_name,
            kernel_size=KERNEL_SIZE,
            wandb_config=wandb_config,
            sample_size=SAMPLE_SIZE,
            is_mul_layer=False,
            has_dropout=False,
            subnet_starting_point=SUBNET_STARTING_POINNT,
            annealing_param=ANNEALING_RATIO,
            model_type=MODEL_TYPE,
            experiment_type=EXPERIENT_TYPE,
            has_inception=HAS_INCEPTION,
        )

        if TEST_RUN:
            break

    # モデルの保存名の書き込み
    JB.dump(
        keys=[
            ENN_TAG,
            DATA_TYPE,
            FIT_POS,
            f"stride_{str(STRIDE)}",
            f"kernel_{str(KERNEL_SIZE)}",
        ],
        value=date_id_saving_list,
    )
