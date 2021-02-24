# ================================================ #
# *            ライブラリのインポート
# ================================================ #

from my_setting import *
SetsPath().set()

# ================================================ #
#     *被験者の名前などのリストを生成するクラス
# ================================================ #

class SubjectsList(object):
    """被験者のリスト

    Args:
        object ([type]): [description]
    """
    def __init__(self) -> None:
        self.nameList = ["H_Li", 
                         "H_Murakami", 
                         "H_Yamamoto",  
                         "H_Kumazawa",
                         "H_Hayashi", 
                         "H_Kumazawa_F", 
                         "H_Takadama", 
                         "H_Hiromoto", 
                         "H_Kashiwazaki"]
        self.ssList = ["WAKE", "REM", "NR1", "NR2", "NR34"]
# ================================================ #
# *            試験用メイン関数
# ================================================ #

if __name__ == '__main__':
    m_subjectsList = SubjectsList()
    print(m_subjectsList.nameList)