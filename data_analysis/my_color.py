from pre_process.json_base import JsonBase


class MyColor(JsonBase):
    def __init__(self) -> None:
        super().__init__(json_filename="my_color.json")
        self.color = self.json_dict["color"]


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
