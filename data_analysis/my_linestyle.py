from dataclasses import dataclass


@dataclass
class MyLinestyle:
    jissen: str = "-"
    hasen: str = "--"
    ittensasen: str = "-."
    tensen: str = ":"


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # 平行線による試し書き
    ls = MyLinestyle()

    for counter, (alias, ls) in enumerate(ls.__dict__.items()):
        if not callable(ls):
            x = np.arange(10)
            # y = x * (counter + 1)
            y = [counter for _ in x]
            plt.plot(x, y, ls=ls, label=f"alias : {alias}")
    plt.legend()
    plt.show()
