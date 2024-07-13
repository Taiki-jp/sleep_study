from dataclasses import dataclass


@dataclass
class MyMarker:
    # jissen: str = "-"
    # hasen: str = "--"
    # ittensasen: str = "-."
    # tensen: str = ":"
    tenmarker: str = "."
    picell_marker: str = ","
    circle: str = "o"
    triangle_down: str = "v"
    triangle_up: str = "^"
    triangle_left: str = "<"
    triangle_right: str = ">"
    sansa_down: str = "1"
    sansa_up: str = "2"
    sansa_left: str = "3"
    sansa_right: str = "4"
    square: str = "s"
    polygon: str = "p"
    star: str = "*"
    hexagon: str = "h"
    hexagon_beta: str = "H"
    plus: str = "+"
    batsu: str = "x"
    diamond: str = "D"
    diamond_tiny: str = "d"
    vertical: str = "|"
    horizontal: str = "_"


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # 平行線による試し書き
    my_marker = MyMarker()

    for counter, (alias, marker) in enumerate(my_marker.__dict__.items()):
        if not callable(marker):
            x = np.arange(10)
            # y = x * (counter + 1)
            y = [counter for _ in x]
            plt.plot(x, y, marker=marker, label=f"alias : {alias}")
    plt.legend()
    plt.show()
