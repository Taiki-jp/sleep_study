from nn.utils import load_model, separate_unc_data
from mywandb.utils import make_ss_dict4wandb
import os
import tensorflow as tf
from data_analysis.utils import Utils
import datetime
import wandb
from pre_process.pre_process import PreProcess
from pre_process.json_base import JsonBase


def main(
    name: str,
    project: str,
    train: list,
    test: list,
    pre_process: PreProcess,
    utils: Utils,
    my_tags: list = None,
    n_class: int = 5,
    test_name: str = None,
    date_id: dict = dict(),
    sample_size: int = 0,
    wandb_config: dict = dict(),
    batch_size: int = 0,
    unc_threthold: float = 0,
    saving_date_id: str = "",
    log_all_in_one: bool = False,
):

    # データセットの作成
    (_, _), (x_test, y_test) = pre_process.make_dataset(
        train=train,
        test=test,
        is_storchastic=False,
        pse_data=False,
        to_one_hot_vector=False,
        each_data_size=sample_size,
    )

    # config の追加
    wandb_config.update(make_ss_dict4wandb(ss_array=y_test, is_train=False))

    # wandbの初期化
    wandb.init(
        name=name,
        project=project,
        tags=my_tags,
        config=wandb_config,
        sync_tensorboard=True,
        dir=pre_process.my_env.project_dir,
    )

    # データクレンジングを行うベースとなるモデルを読み込む
    model = load_model(
        loaded_name=test_name,
        n_class=n_class,
        verbose=1,
        model_id=date_id,
        is_positive=False,
        is_negative=False,
    )
    # データがキャッチできていない場合はreturn
    if model is None:
        return
    # ポジティブクレンジングを行ったモデルを読み込む
    positive_model = load_model(
        loaded_name=test_name,
        n_class=n_class,
        verbose=1,
        model_id=date_id,
        is_positive=True,
        is_negative=False,
    )
    # データがキャッチできていない場合はreturn
    if positive_model is None:
        return

    # テストデータのクレンジング
    (_x_test, _y_test) = separate_unc_data(
        x=x_test,
        y=y_test,
        model=model,
        batch_size=batch_size,
        n_class=n_class,
        experiment_type="positive_cleansing",
        unc_threthold=unc_threthold,
        verbose=0,
    )

    # 各睡眠段階のF値を計算wandbに送信
    # Utils().calc_ss_acc(
    #     x=_x_test,
    #     y=_y_test,
    #     model=model,
    #     n_class=n_class,
    #     batch_size=batch_size,
    # )

    # # クレンジング後のデータに対してグラフを作成
    # 不確実性の高いデータのみで一致率を計算
    evidence_base = model.predict(_x_test, batch_size=batch_size)
    evidence_positive = positive_model.predict(_x_test)

    # 睡眠段階の予測
    _, _, _, y_pred_base = utils.calc_enn_output_from_evidence(evidence=evidence_base)
    _, _, _, y_pred_pos = utils.calc_enn_output_from_evidence(evidence=evidence_positive)
    # 一致率の計算
    acc_base = utils.calc_acc_from_pred(y_true = _y_test.numpy(), y_pred=y_pred_base, log_label="base")
    acc_sub = utils.calc_acc_from_pred(y_true = _y_test.numpy(), y_pred=y_pred_pos, log_label="sub")
    # csv出力
    # output_path = "20211018_for_box_plot.csv"
    # utils.to_csv(df_result, path=output_path, edit_mode="append")

    # wandb終了
    wandb.finish()


if __name__ == "__main__":
    # 環境設定
    CALC_DEVICE = "gpu"
    DEVICE_ID = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID
    tf.keras.backend.set_floatx("float32")
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # ANCHOR: ハイパーパラメータの設定
    TEST_RUN = False
    WANDB_PROJECT = "2021_code00"
    IS_MUL_LAYER = False
    CATCH_NREM2 = True
    BATCH_SIZE = 128
    N_CLASS = 5
    STRIDE = 480
    SAMPLE_SIZE = 5000
    UNC_THRETHOLD = 0.5
    EXPERIMENT_TYPES = (
        "no_cleansing",
        "positive_cleansing",
        "negative_cleansing",
    )

    # オブジェクトの作成
    pre_process = PreProcess(
        data_type="spectrum",
        fit_pos="middle",
        verbose=0,
        kernel_size=512,
        is_previous=False,
        stride=STRIDE,
        is_normal=True,
    )
    datasets = pre_process.load_sleep_data.load_data(
        load_all=True, pse_data=False
    )
    utils = Utils(catch_nrem2=CATCH_NREM2)

    # 読み込むモデルの日付リストを返す
    JB = JsonBase("../nn/model_id.json")
    JB.load()
    model_date_list = JB.make_list_of_dict_from_mul_list(
        "enn", "spectrum", "middle", f"stride_{STRIDE}", "kernel_512"
    )

    # モデルのidを記録するためのリスト
    date_id_saving_list = list()

    for test_id, (test_name, date_id) in enumerate(
        zip(pre_process.name_list, model_date_list)
    ):
        (train, test) = pre_process.split_train_test_from_records(
            datasets, test_id=test_id
        )

        # 保存用の時間id
        saving_date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # date_id_saving_list.append(saving_date_id)

        # tagの設定
        my_tags = [
            test_name,
        ]
        wandb_config = {
            "test name": test_name,
        }
        # FIXME: name をコード名にする
        main(
            name=f"{test_name}",
            project=WANDB_PROJECT,
            train=train,
            test=test,
            pre_process=pre_process,
            my_tags=my_tags,
            date_id=date_id,
            test_name=test_name,
            n_class=N_CLASS,
            wandb_config=wandb_config,
            batch_size=BATCH_SIZE,
            unc_threthold=UNC_THRETHOLD,
            saving_date_id=saving_date_id,
            log_all_in_one=True,
            sample_size=SAMPLE_SIZE,  # これがないとtrainの分割のところでデータがなくてエラーが起きてしまう
            utils=utils,
        )

        # testの時は一人の被験者で止める
        if TEST_RUN:
            break
