import os
import tensorflow as tf
from data_analysis.utils import Utils
from pre_process.pre_process import PreProcess
from nn.utils import load_model, separate_unc_data

# seedの固定
tf.random.set_seed(100)
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


TEST_RUN = False
HAS_ATTENTION = True
PSE_DATA = False
HAS_INCEPTION = True
IS_PREVIOUS = False
IS_NORMAL = True
IS_ENN = True  # FIXME: always true so remove here
IS_MUL_LAYER = False
CATCH_NREM2 = True
EPOCHS = 200
BATCH_SIZE = 16
N_CLASS = 5
KERNEL_SIZE = 512
STRIDE = 480
SAMPLE_SIZE = 1000
UNC_THRETHOLD = 0.5
DATA_TYPE = "spectrum"
FIT_POS = "middle"
EXPERIMENT_TYPES = (
    "no_cleansing",
    "positive_cleansing",
    "negative_cleansing",
)
EXPERIENT_TYPE = "positive_cleansing"
NORMAL_TAG = "normal" if IS_NORMAL else "sas"
ATTENTION_TAG = "attention" if HAS_ATTENTION else "no-attention"
PSE_DATA_TAG = "psedata" if PSE_DATA else "sleepdata"
INCEPTION_TAG = "inception" if HAS_INCEPTION else "no-inception"
WANDB_PROJECT = "test" if TEST_RUN else "enn4fixed_stride_fixed_sample"
ENN_TAG = "enn" if IS_ENN else "dnn"
INCEPTION_TAG += "v2" if IS_MUL_LAYER else ""
CATCH_NREM2_TAG = "catch_nrem2" if CATCH_NREM2 else "catch_nrem34"


# モデルの読み込み
loaded_name = "140703_Li"
date_id = {
    "nothing": "20211010-150603",
    "negative": "20211010-150603",
    "positive": "20211010-150603",
}
n_class = 5
model = load_model(
    loaded_name=loaded_name, model_id=date_id, n_class=n_class, verbose=0
)
# model.summary()

# Recordオブジェクトの読み込み
pre_process = PreProcess(
    data_type=DATA_TYPE,
    fit_pos=FIT_POS,
    verbose=0,
    kernel_size=KERNEL_SIZE,
    is_previous=IS_PREVIOUS,
    stride=STRIDE,
    is_normal=IS_NORMAL,
    has_nrem2_bias=True,
)
datasets = pre_process.load_sleep_data.load_data(
    load_all=True, pse_data=PSE_DATA
)
utils = Utils(catch_nrem2=CATCH_NREM2)

# 訓練データとテストデータの分類
(train, test) = pre_process.split_train_test_from_records(
    datasets, test_id=0, pse_data=PSE_DATA
)
(x_train, y_train), (x_test, y_test) = pre_process.make_dataset(
    train=train,
    test=test,
    is_storchastic=False,
    pse_data=PSE_DATA,
    to_one_hot_vector=False,
    each_data_size=SAMPLE_SIZE,
)

# 訓練データのクレンジング
(_x, _y) = separate_unc_data(
    x=x_train,
    y=y_train,
    model=model,
    batch_size=BATCH_SIZE,
    n_class=n_class,
    experiment_type=EXPERIENT_TYPE,
    unc_threthold=UNC_THRETHOLD,
    verbose=0,
)
