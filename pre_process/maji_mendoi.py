import pandas as pd

class MajiMendoi(object):
    def __init__(self) -> None:
        self.df_list = list()
        pass
    
    def set_kusomendoi(self):
        #
        #wake, rem, nr1, nr2, nr34の順番になるように

        # Liさんブロック
        df_li = list()
        df_li.append(pd.DataFrame([[16,26], [25,15]]))
        df_li.append(pd.DataFrame([[31,55], [134,110]]))
        df_li.append(pd.DataFrame([[7,11],[30,26]]))
        df_li.append(pd.DataFrame([[108,133],[287,262]]))
        df_li.append(pd.DataFrame([[5,6],[150,149]]))
        
        # Hiromotoさんブロック
        df_hiromoto = list()
        df_hiromoto.append(pd.DataFrame([[0,20],[32,12]]))
        df_hiromoto.append(pd.DataFrame([[19,54],[215,180]]))
        df_hiromoto.append(pd.DataFrame([[0,0],[51,51]]))
        df_hiromoto.append(pd.DataFrame([[65,58],[423,430]]))
        df_hiromoto.append(pd.DataFrame([[0,0],[0,0]]))
        
        # Kumazawaさんブロック
        df_kumazawa = list()
        df_kumazawa.append(pd.DataFrame([[8,40],[62,30]]))
        df_kumazawa.append(pd.DataFrame([[5,10],[56,51]]))
        df_kumazawa.append(pd.DataFrame([[3,8],[41,36]]))
        df_kumazawa.append(pd.DataFrame([[32,102],[358,288]]))
        df_kumazawa.append(pd.DataFrame([[0,0],[2,2]]))
        
        # Kumazawa_Fさんブロック
        df_kumazawa_f = list()
        df_kumazawa_f.append(pd.DataFrame([[17,25],[15,7]]))
        df_kumazawa_f.append(pd.DataFrame([[2,12],[112,102]]))
        df_kumazawa_f.append(pd.DataFrame([[5,4],[30,31]]))
        df_kumazawa_f.append(pd.DataFrame([[8,19],[203,192]]))
        df_kumazawa_f.append(pd.DataFrame([[28,49],[42,21]]))
        
        # Kashiwazakiさんブロック
        df_kashiwazaki = list()
        df_kashiwazaki.append(pd.DataFrame([[5,65],[86,26]]))
        df_kashiwazaki.append(pd.DataFrame([[57,100],[91,48]]))
        df_kashiwazaki.append(pd.DataFrame([[0,1],[25,24]]))
        df_kashiwazaki.append(pd.DataFrame([[31,72],[378,337]]))
        df_kashiwazaki.append(pd.DataFrame([[0,0],[0,0]]))
        
        # Hayashiさんブロック
        df_hayashi = list()
        df_hayashi.append(pd.DataFrame([[2,13],[39,28]]))
        df_hayashi.append(pd.DataFrame([[0,0],[79,79]]))
        df_hayashi.append(pd.DataFrame([[0,7],[49,42]]))
        df_hayashi.append(pd.DataFrame([[172,234],[193,131]]))
        df_hayashi.append(pd.DataFrame([[0,0],[22,22]]))
        
        # Takadamaさんブロック
        df_Takadama = list()
        df_Takadama.append(pd.DataFrame([[2,12],[46,36]]))
        df_Takadama.append(pd.DataFrame([[9,15],[95,89]]))
        df_Takadama.append(pd.DataFrame([[15,13],[61,63]]))
        df_Takadama.append(pd.DataFrame([[38,82],[342,298]]))
        df_Takadama.append(pd.DataFrame([[0,0],[1,1]]))
        
        # Murakamiさんブロック
        df_murakami = list()
        df_murakami.append(pd.DataFrame([[3,26],[46,23]]))
        df_murakami.append(pd.DataFrame([[7,38],[99,68]]))
        df_murakami.append(pd.DataFrame([[2,3],[43,42]]))
        df_murakami.append(pd.DataFrame([[17,41],[328,304]]))
        df_murakami.append(pd.DataFrame([[0,0],[1,1]]))
        
        # Yamamotoさんブロック
        df_yamamoto = list()
        df_yamamoto.append(pd.DataFrame([[0,5],[95,90]]))
        df_yamamoto.append(pd.DataFrame([[1,12],[172,161]]))
        df_yamamoto.append(pd.DataFrame([[1,1],[32,32]]))
        df_yamamoto.append(pd.DataFrame([[3,27],[354,330]]))
        df_yamamoto.append(pd.DataFrame([[0,0],[0,0]]))
        
        # subjectsListの名前通りにappendする
        tmp_list = [df_li, df_murakami, df_yamamoto, 
                    df_kumazawa, df_hayashi,df_kumazawa_f,
                    df_Takadama, df_hiromoto, df_kashiwazaki]
        for df in tmp_list:
            self.df_list.append(df)
        assert len(self.df_list) == 9
        assert len(self.df_list[0]) == 5


if __name__ == "__main__":
    from maji_mendoi import MajiMendoi
    from utils import Utils
    import pandas as pd
    import os
    obj = MajiMendoi()
    obj.set_kusomendoi()
    #print(obj.df_list)
    tmp = obj.df_list
    #df_cm = pd.DataFrame(tmp)
    o_utils = Utils()
    prop_dict = o_utils.show_properties(tmp)
    prop_df = pd.DataFrame(prop_dict)
    path = os.path.join(os.environ['sleep'], "datas", "property.csv")
    prop_df.to_csv(path)
    print("stop")
    
           
    