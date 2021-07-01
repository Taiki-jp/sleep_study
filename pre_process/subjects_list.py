import sys, os

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
        self.sas_name_list = ["sas_5993",
                              "sas_30929",
                              "sas_33792",
                              "sas_35818",
                              "sas_35866",
                              "sas_35997",
                              "sas_36162",
                              "sas_36350",
                              "sas_36765"]
        # 追加分の健常者
        self.added_name_list = os.path.join(os.environ["sleep"],
                                            "datas",
                                            "adding_sleep_datas")
        # 追加分のSAS
        self.added_sas_name_list = os.path.join(os.environ["sleep"],
                                                "datas",
                                                "sas_datasets")
                                            
        self.name_dict = None
        self.spectrogram_date = "20210201-055748"  # スペクトログラム
        # previous : "20210320-011750"
        # middle(16) : "20210630-015133"
        # top(16) : 20210630-015349
        # bottom(16) : 20210630-014416
        # middle(1) : "20210630-024223"
        # top(1) : "20210630-023745"
        # bottom(1) : "20210630-024629"
        # bottom(4) : "20210701-145039"
        # middle(4) : "20210701-183500"
        # ? : "20210701-165511"
        # top(4) : "20210701-203340"
        self.spectrum_date = "20210630-032737"  #   # スペクトラム版
        # attn x incpt x spc_2d
        self.date_id_list_attnt_incpt_spc_2d = ["20210601-051642",
                                                "20210601-053406",
                                                "20210602-120441",   #yamamoto : "20210601-055045",
                                                "20210601-060739",
                                                "20210601-062423",
                                                "20210601-064055",
                                                "20210601-065654",
                                                "20210602-165042"] #hiromoto : ""20210601-071330",
        # attn x incpt x spc_1d
        self.date_id_list_attnt_incpt_spc_1d = ["20210608-144244",
                                                "20210608-151256",
                                                "20210608-154059",
                                                "20210608-160954",
                                                "20210608-163811",
                                                "20210608-170631",
                                                "20210608-173335",
                                                "20210608-180219",
                                                "20210608-183218"]
        # attn x spc_2d
        self.date_id_list_attnt_spc_2d = ["20210604-070048",
                                          "20210604-071219",
                                          "20210604-072311",
                                          "20210604-073420",
                                          "20210604-074528",
                                          "20210604-075638",
                                          "20210604-080726",
                                          "20210604-081841",
                                          "20210604-083019"]
        # attn x spc_1d
        self.date_id_list_attnt_spc_1d = list()
        # incpt x spc_2d
        self.date_id_list_incpt_spc_2d = list()
        # incpt x spec_1d
        self.date_id_list_incpt_spc_1d = list()
        # spec_2d
        self.date_id_list_spc_2d = list()
        # spec_1d
        self.date_id_list_spc_1d = list()

    

    
    def sets_filename(self, data_type, n_class):
        if data_type == "spectrum":
            self.name_dict = {name : name+"_"+self.spectrum_date+".sav" for name in self.name_list}
        elif data_type == "spectrogram":
            self.name_dict = {name : name+"_"+self.spectrogram_date+".sav" for name in self.name_list}
        else:
            print(f"data_typeを正しく指定してください")
            sys.exit(1)
        if n_class == 5:
            self.ss_list = ["nr34", "nr2", "nr1", "rem", "wake"]
        elif n_class == 4:
            self.ss_list = ["nr34", "nr12", "rem", "wake"]
        elif n_class == 3:
            self.ss_list = ["nr", "rem", "weke"]
        elif n_class == 2:
            self.ss_list = ["non-target", "target"]
        else:
            print("正しいクラス数を指定してください")
            sys.exit(1)
if __name__ == '__main__':
    sl = SubjectsList()
    print("name_list : ", sl.name_list)
    print("name_dict(before set) : ", sl.name_dict)
    # セット(スペクトログラム)
    sl.sets_filename(data_type="spectrogram", n_class=5)
    print("name_dict(set spectrogram) : ", sl.name_dict)
    # セット（スペクトラム）
    sl.sets_filename(data_type="spectrum", n_class=5)
    print("name_dict(set spectrum) : ", sl.name_dict)
    