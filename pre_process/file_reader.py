import pickle, sys, os
from pre_process.subjects_list import SubjectsList
from pre_process.record import Record
from pre_process.my_setting import SetsPath
SetsPath().set()
import record

class FileReader(object):
    def __init__(self, n_class):
        self.sl = SubjectsList()
        # データの保存先（パスを指定） NOTE : 本当は絶対パスの設定は一か所にまとめたいけど，
        # 現在の実装ではutilsを循環参照してしまうので，仕方なく下で定義
        self.dir_name = os.path.join(os.environ['sleep'], "datas", "pre_processed_data")
        self.n_class = n_class

    def load_normal(self, name, verbose=0, data_type=None, fit_pos="middle"):
        # ファイル名が指定されておらず、データ型が与えられたときはsetメソッドを呼ぶ
        if data_type is not None and self.sl.added_name_dict==None:
            self.sl.sets_filename(data_type=data_type, n_class=self.n_class)
        file_name = self.sl.added_name_dict[name]
        path = os.path.join(self.dir_name,
                            data_type,
                            fit_pos,
                            file_name)
        if verbose==0:
            print(f"{path}を読み込んでいます...")
        elif verbose==1:
            print(f"入力の種類は{data_type}です．被験者{name}を読み込んでいます...")
        else:
            pass
        return pickle.load(open(path, 'rb'))

if __name__ == '__main__':
    fr = FileReader(n_class=5)
    data = fr.load_normal(name="H_Li", data_type="spectrum", verbose=1, fit_pos="middle")
    print("data_len : ", len(data[0]))
    