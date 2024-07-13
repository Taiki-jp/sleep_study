from __future__ import annotations

import os
import random
import sys
from typing import Dict, List

import numpy as np
import tensorflow as tf
from numpy.lib.function_base import kaiser
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.keras.engine.training import Model

from data_analysis.py_color import PyColor
from nn.losses import EDLLoss

# 2クラス分類のモデル読み込み用関数


def load_bin_model(
    loaded_name: str,
    verbose: int,
    is_all: bool,
    ss_id: str,
    ss: str = "",
) -> List[tf.keras.Model]:
    # 表示するかどうか
    if verbose != 0:
        print(PyColor.GREEN, f"*** {loaded_name}のモデルを読み込みます ***", PyColor.END)

    # モデルの読み込みメソッド
    def _load_model(ss):
        # denn_ensemble.pyの時にss_idがstr型で来るため修正した
        filepath = os.path.join(
            os.environ["sleep"], "models", loaded_name, ss, ss_id
        )
        if not os.path.exists(filepath):
            print(PyColor.RED_FLASH, f"{filepath}は存在しません", PyColor.END)
            sys.exit(1)
        print(PyColor.GREEN_FLASH, f"{filepath}を読み込みます", PyColor.END)
        model = tf.keras.models.load_model(
            filepath, custom_objects={"EDLLoss": EDLLoss(K=2, annealing=0.1)}
        )
        return model

    if is_all:
        tmp = list()
        print(PyColor.GREEN, "全ての睡眠段階のモデルを読み込みます", PyColor.END)
        for ss in ["nr1", "nr2", "nr3", "rem", "wake"]:
            tmp.append(_load_model(ss=ss))
    else:
        tmp = list()
        print(PyColor.GREEN, f"{ss}の睡眠段階のモデルを読み込みます", PyColor.END)
        tmp.append(_load_model(ss))
    return tmp


# モデルの読み込み用関数
def load_model(
    loaded_name: str,
    n_class: int,
    verbose: int,
    model_id: Dict[str, str] | str = None,
    is_positive: bool = False,
    is_negative: bool = False,
) -> Model:
    # 表示するかどうか
    if not isinstance(model_id, dict):
        path = os.path.join(
            os.environ["sleep"],
            "models",
            loaded_name,
            model_id,
        )
    else:
        if verbose != 0:
            print(
                PyColor.GREEN, f"*** {loaded_name}のモデルを読み込みます ***", PyColor.END
            )
        # positive かつ negative でない
        if is_positive and is_negative == False:
            path = os.path.join(
                os.environ["sleep"],
                "models",
                loaded_name,
                model_id["positive"],
            )
        # negative かつ positive でない
        elif is_negative and is_positive == False:
            path = os.path.join(
                os.environ["sleep"],
                "models",
                loaded_name,
                model_id["negative"],
            )
        # 上に漏れた場合は no-cleansing を読み込む
        else:
            path = os.path.join(
                os.environ["sleep"],
                "models",
                loaded_name,
                model_id["nothing"],
            )
    if not os.path.exists(path):
        print(PyColor.RED_FLASH, f"{path}は存在しません", PyColor.END)
        return None
    model = tf.keras.models.load_model(
        path, custom_objects={"EDLLoss": EDLLoss(K=n_class, annealing=0.1)}
    )
    if verbose != 0:
        print(PyColor.GREEN, f"*** {loaded_name}のモデルを読み込みました ***", PyColor.END)
    return model


# 不確かさをもとにデータを分類する関数
def separate_unc_data(
    x: Tensor,
    y: Tensor,
    model: Model,
    batch_size: int,
    n_class: int,
    experiment_type: str,
    unc_threthold: float,
    verbose: int,
) -> tuple:
    evidence = model.predict(x, batch_size=batch_size)
    alpha = evidence + 1
    unc = n_class / tf.reduce_sum(alpha, axis=1, keepdims=True)
    if experiment_type == "positive_cleansing":
        mask = unc > unc_threthold
    elif experiment_type == "negative_cleansing":
        mask = unc < unc_threthold
    else:
        raise Exception("正しい実験タイプを指定してください")
    if verbose != 0:
        print(
            PyColor.GREEN, f"*** {experiment_type}でデータを整えます ***", PyColor.END
        )
    return (
        tf.boolean_mask(x, mask.numpy().reshape(x.shape[0])),
        tf.boolean_mask(y, mask.numpy().reshape(x.shape[0])),
    )


def set_seed(seed=200):
    tf.random.set_seed(seed)
    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
