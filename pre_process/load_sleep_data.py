from pre_process.file_reader import FileReader

class LoadSleepData():

    def __init__(self, data_type, verbose=0):
        self.fr = FileReader()
        # NOTE : filereaderのsubjects_listを持ってくること
        self.sl = self.fr.sl
        # NOTE : オブジェクト生成時にデータの種類を決める（ファイル名が決まる）
        self.data_type = data_type
        # data_typeが決まった時点で読み込むファイル名は決まるのでここでsets_filenameを行う
        self.sl.sets_filename(data_type=self.data_type)
        self.verbose = verbose
    
    def load_data(self, name=None, load_all=False):
        if load_all:
            print("*** すべての被験者を読み込みます ***")
            records = list()
            for name in self.sl.name_list:
                records.extend(self.fr.load_normal(name=name, 
                                                   verbose=self.verbose, 
                                                   data_type=self.data_type))
            return records        
        else:
            print("*** 一人の被験者を読み込みます ***")
            return self.fr.load_normal(name=name,
                                       verbose=self.verbose,
                                       data_type=self.data_type)

if __name__ == "__main__":

    # 確率的に各被験者からデータを取ってくる
    import numpy as np
    from collections import Counter
    import random
    
    load_sleep_data = LoadSleepData(data_type="spectrogram", verbose=1)
    data = load_sleep_data.load_data(name="H_Li", load_all=False)
    # or
    data = load_sleep_data.load_data(load_all=True)
    print("data_len : ", len(data))
    