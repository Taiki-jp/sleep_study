import tensorflow as tf
from data_analysis.py_color import PyColor
import os
import sys
from tensorflow.python.keras.engine.training import Model
from nn.losses import EDLLoss

# loaded_name => test_name, model_id => model_id, n_class => n_class
def load_model(
    loaded_name: str, model_id: str, n_class: int, verbose: int
) -> Model:
    if verbose != 0:
        print(PyColor.GREEN, f"*** {loaded_name}のモデルを読み込みます ***", PyColor.END)
    path = os.path.join(os.environ["sleep"], "models", loaded_name, model_id)
    if not os.path.exists(path):
        print(PyColor.RED_FLASH, f"{path}は存在しません", PyColor.END)
        sys.exit(1)
    model = tf.keras.models.load_model(
        path, custom_objects={"EDLLoss": EDLLoss(K=n_class, annealing=0.1)}
    )
    if verbose != 0:
        print(PyColor.GREEN, f"*** {loaded_name}のモデルを読み込みました ***", PyColor.END)
    return model
