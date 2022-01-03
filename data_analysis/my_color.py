from pre_process.json_base import JsonBase


class MyColor(JsonBase):
    def __init__(self) -> None:
        super().__init__(json_filename="my_color.json")
        self.color = self.json_dict["color"]
        print(self.color)


if __name__ == "__main__":
    my_color = MyColor()
    # my_color.load()
    # for counter, (alias, color) in enumerate(my_color.json_dict.items()):
    #     if not callable(color):
    #         x = np.arange(10)
    #         y = x * (counter + 1)
    #         plt.plot(x, y, c=color, label=f"alias : {alias}")
    # plt.legend()
    # plt.show()
