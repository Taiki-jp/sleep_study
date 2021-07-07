from pre_process.my_setting import FindsDir
import pandas as pd
import os


class CsvReader(FindsDir):
    def __init__(self, target_problem, load_dir, person_dir, filename):
        super().__init__(target_problem=target_problem)
        self.root_dirname = self.return_filepath()
        self.load_dir = load_dir
        self.personDirName = person_dir
        self.fileName = filename
        self.df = None

    def read_csv(self):
        filePath = os.path.join(
            self.rootDirName, self.loadDir, self.personDirName, self.fileName
        )

        def _read():
            print(f"*** read {filePath} ***")
            self.df = pd.read_csv(filePath)

        return _read()


if __name__ == "__main__":

    csv_reader = CsvReader(
        "sleep", "datas/my_raw_data", "H_Hayashi", "signal_after.csv"
    )
    csv_reader.readCsv()
