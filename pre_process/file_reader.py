import pickle, sys, os
from pre_process.subjects_list import SubjectsList
import pre_process.record as record

class FileReader(object):
    def __init__(self):
        self.sl = SubjectsList()
        # データの保存先（パスを指定）
        self.dir_name = os.path.join(os.environ['sleep'], "datas", "pre_processed_data")

    def load_normal(self, name, verbose=0, data_type=None):
        if data_type is not None:
            # data_typeが指定されたときのみセットを行う（これを行わないとエラーになる）
            self.sl.sets_filename(data_type=data_type)
        file_name = self.sl.name_dict[name]
        path = os.path.join(self.dir_name, file_name)
        print(path)
        return pickle.load(open(path, 'rb'))

if __name__ == '__main__':
    print(globals())
    fr = FileReader()
    data = fr.load_normal(name="H_Li", data_type="spectrogram")
    print("data_len", len(data[0]))
    # FIXME : モジュールとして実行するとできないけど，F5実行するとできる
    