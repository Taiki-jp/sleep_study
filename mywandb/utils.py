from collections import Counter
from tensorflow.python.framework.ops import Tensor
from numpy import ndarray

# 各睡眠段階の数を返す関数
def make_ss_dict4wandb(ss_array: Tensor, is_train: bool) -> dict:
    try:
        assert type(ss_array) == Tensor or type(ss_array) == ndarray
    except:
        raise AssertionError("ss_array の型チェックしてください")
    if type(ss_array) == Tensor:
        d = Counter(ss_array.numpy())
    else:
        d = Counter(ss_array)
    ss_labels = ["num_nr34", "num_nr2", "num_nr1", "num_rem", "num_wake"]
    if is_train:
        post_key = "_train"
    else:
        post_key = "_test"
    # 存在する睡眠段階のみ返す
    dd = {ss_labels[i] + post_key: d[i] for i in d.keys()}
    return dd
