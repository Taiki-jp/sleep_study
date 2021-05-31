import sys

# NOTE : ここ以外で被験者の名前のリストを作成しない
class SubjectsList(object):
    
    def __init__(self) -> None:
        self.name_list = ["H_Li", 
                         "H_Murakami", 
                         "H_Yamamoto",  
                         "H_Kumazawa",
                         "H_Hayashi", 
                         "H_Kumazawa_F", 
                         "H_Takadama", 
                         "H_Hiromoto", 
                         "H_Kashiwazaki"]
        self.name_dict = None
        self.spectrogram_date = "20210201-055748"  # スペクトログラム
        self.spectrum_date = "20210320-011750"  # スペクトラム版
        self.ss_list = ["nr34", "nr2", "nr1",
                        "rem", "wake"]
        
    def sets_filename(self, data_type):
        if data_type == "spectrum":
            self.name_dict = {name : name+"_"+self.spectrum_date+".sav" for name in self.name_list}
            return
        elif data_type == "spectrogram":
            self.name_dict = {name : name+"_"+self.spectrogram_date+".sav" for name in self.name_list}
            return
        else:
            print(f"data_typeを正しく指定してください")
            sys.exit(1)

if __name__ == '__main__':
    sl = SubjectsList()
    print("name_list : ", sl.name_list)
    print("name_dict(before set) : ", sl.name_dict)
    # セット(スペクトログラム)
    sl.sets_filename(data_type="spectrogram")
    print("name_dict(set spectrogram) : ", sl.name_dict)
    # セット（スペクトラム）
    sl.sets_filename(data_type="spectrum")
    print("name_dict(set spectrum) : ", sl.name_dict)
    