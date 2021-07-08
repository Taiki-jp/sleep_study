from dataclasses import dataclass
import time


@dataclass
class PyColor:
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    PURPLE = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RETURN = "\033[07m"  # 反転
    ACCENT = "\033[01m"  # 強調
    FLASH = "\033[05m"  # 点滅
    RED_FLASH = "\033[05;41m"  # 赤背景+点滅
    BOLD = "\033[1m"  # 太字
    END = "\033[0m"


if __name__ == "__main__":
    my_color = PyColor()
    print(my_color.FLASH + "hoge" + my_color.END)
    print(my_color.BLUE, "hoge", my_color.END)
    print(my_color.BLUE, my_color.FLASH, "hoge", my_color.END)
    print(my_color.BLUE, my_color.RETURN, "hoge", my_color.END)
    print(my_color.RED_FLASH, "hoge", my_color.END)
    print(my_color.CYAN, my_color.BOLD, "hoge", my_color.END)
    time.sleep(10)
