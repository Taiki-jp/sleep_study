import pickle, sys, os
from pre_process.subjects_list import SubjectsList
from pre_process.record import Record
from pre_process.my_env import MyEnv
from data_analysis.py_color import PyColor
from pre_process.pre_processed_id import PreProcessedId

class FileReader(object): 
    def __init__(self) -> None:

        self.sl : SubjectsList = SubjectsList()
        self.sl.load()
        self.my_env : MyEnv = MyEnv()
        self.ppi = PreProcessedId()
        self.ppi.load()
        self.sl.load()

    # ファイルを読み込むためにファイルパスと被験者名を指定する
    def load(self, 
             name : str = "",
             path_list : list = [],
             verbose : int = 0):

        # nameが空のときはエラーハンドル
        if name == "":
            print(PyColor.RED_FLASH,
                  "nameを指定してください",
                  PyColor.END)
            sys.exit(1)
        
        filepath = self.my_env.pre_processed_dir
        filepath= os.path.join(filepath, *path_list)
        
        def _load():

            if verbose==0:
                print(PyColor.YELLOW,
                      f"{filepath}を読み込んでいます...",
                      PyColor.END)
            elif verbose==1:
                print(PyColor.YELLOW,
                      f"被験者{name}を読み込んでいます...",
                      PyColor.END)
            else:
                pass
            return pickle.load(open(filepath, 'rb'))
            
        return _load()

if __name__ == '__main__':
    # 既存のデータは仕方なくこれでインポートが必要
    import sys
    sys.path.append("c:/users/takadamalab/git/sleep_study/pre_process")
    from record import Record
    fr = FileReader()
    data01 = fr.load(name = "H_Li", 
                   path_list = [
                       "previous_dataset",
                       "H_Li_"+fr.ppi.prev_datasets["spectrum"]["id"]+".sav",
                   ])
    data02 = fr.load(name = "140703_Li",
                     path_list = [
                         "spectrum",
                         "middle",
                         "140703_Li_"+fr.ppi.spectrum["middle"]["stride_16"]+".sav"
                     ])
    datas = [data01, data02]
    for data in datas:
        print("data_len : ", len(data[0]))
    