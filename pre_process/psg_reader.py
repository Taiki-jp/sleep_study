from pre_process.csv_reader import CsvReader
import pandas as pd
import os


class PsgReader(CsvReader):
    def __init__(
        self,
        person_dir: str,
        file_name: str = "sleepStage.csv",
        verbose: int = 0,
        is_previous: bool = True,
        columns: list = [0, 2],
        names: list = ["time", "ss"],
    ):

        super().__init__(
            person_dir=person_dir,
            file_name=file_name,
            verbose=verbose,
            columns=columns,
            names=names,
        )
        if not is_previous:
            self.file_name = "_" + file_name
            self.columns = [1, 2]


if __name__ == "__main__":
    psg = PsgReader("H_Hayashi")
    psg.read_csv(columns=[1, 3], names=["time", "ss"])
