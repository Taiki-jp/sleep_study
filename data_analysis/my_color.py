from dataclasses import dataclass, asdict, astuple


@dataclass
class MyColor:
    # NOTE ： 型定義をしないと__dict__で取り出せないので注意
    WHITE_BLUE: str = "#43caf4"
    WHITE_PINK: str = "#f44372"
    LIGHT_GRAYISH_BLUE: str = "#dddff3"
    LIGHT_GRAYISH_MAGENTA: str = "#f1ddf3"
    LIGHT_GRAYISH_RED: str = "#f1e2e1"
    LIGHT_GRAYISH_LIME_GREEN: str = "#e1f1e2"
    LIGHT_GRAYISH_YELLOW: str = "#f0f1e1"
    WHITE_ORANGE: str = "#f46d43"
    YELLOW: str = "#f4c643"
    BLUE: str = "#4371f4"
    BLIGHT_CYAN: str = "#43f4c6"
    nrem34: str = "#4372f4"
    nrem2: str = "#f46d43"
    nrem1: str = "#9799a0"
    rem: str = "#f4c643"
    wake: str = "#43caf4"


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # 平行線による試し書き
    my_color = MyColor()

    for counter, (alias, color) in enumerate(my_color.__dict__.items()):
        if not callable(color):
            x = np.arange(10)
            # y = x * (counter + 1)
            y = [counter for _ in x]
            plt.plot(x, y, c=color, label=f"alias : {alias}")
    plt.legend()
    plt.show()
