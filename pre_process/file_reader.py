import pickle, sys, os
from pre_process.subjects_list import SubjectsList
# NOTE : pickleは保存時と読み込み時のパスの指定を同じ方法で書かなければならないみたい
# FIXME : 出来るだけ、pathのappendをしたくないので変更案が出れば修正
from pre_process.my_setting import SetsPath
SetsPath().set()
import record

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
        if verbose==0:
            print(f"{path}を読み込んでいます...")
        elif verbose==1:
            print(f"入力の種類は{data_type}です．被験者{name}を読み込んでいます...")
        else:
            pass
        return pickle.load(open(path, 'rb'))

if __name__ == '__main__':
    fr = FileReader()
    data = fr.load_normal(name="H_Li", data_type="spectrogram")
    print("data_len", len(data[0]))
    