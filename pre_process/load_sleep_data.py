from pre_process.file_reader import FileReader

class LoadSleepData():
    # NOTE : n_class is needed because sets_filename method of file_reader is called in here
    def __init__(self, data_type, verbose=0, n_class=5):
        self.fr = FileReader(n_class=n_class)
        # NOTE : filereaderのsubjects_listを持ってくること
        self.sl = self.fr.sl
        # NOTE : オブジェクト生成時にデータの種類を決める（ファイル名が決まる）
        self.data_type = data_type
        # data_typeが決まった時点で読み込むファイル名は決まるのでここでsets_filenameを行う（基本ここで呼ばれる）
        self.sl.sets_filename(data_type=self.data_type, n_class=n_class)
        self.verbose = verbose
    
    def load_data(self, name=None, load_all=False, pse_data=False,
                  fit_pos = None):
        # NOTE : pse_data is needed for avoiding to load data
        if pse_data:
            print("仮データのため、何も読み込みません")
            return None
        if load_all:
            print("*** すべての被験者を読み込みます（load_dataの引数:nameは無視します） ***")
            records = list()
            for name in self.sl.name_list:
                records.extend(self.fr.load_normal(name=name, 
                                                   verbose=self.verbose, 
                                                   data_type=self.data_type,
                                                   fit_pos=fit_pos))
            return records        
        else:
            print("*** 一人の被験者を読み込みます ***")
            return self.fr.load_normal(name=name,
                                       verbose=self.verbose,
                                       data_type=self.data_type)

if __name__ == "__main__":
    from collections import Counter
    load_sleep_data = LoadSleepData(data_type="spectrum", verbose=0, n_class=5)
    data = load_sleep_data.load_data(load_all=True, pse_data=False, name=None,
                                     fit_pos="bottom")
    # 各被験者について睡眠段階の量をチェック
    for each_data in data:
        ss = [record.ss for record in each_data]
        print(Counter(ss))
    