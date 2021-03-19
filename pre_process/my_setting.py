# ================================================ #
# *            ライブラリのインポート
# ================================================ #

import os, sys

# ================================================ #
# *インポート先のファイルのパスを追加するクラス
# ================================================ #

class SetsPath(object):
    # 一度しか set メソッドを実行しないためのクラス変数
    executedFlag = False
    
    def __init__(self):
        """git/ 以下にあるプログラムのための必要なフォルダをインポートするために設定する
        """
        self.dataAnalysis = os.path.join(os.environ["git"], "sleep_study", "data_analysis")
        self.preProcess = os.path.join(os.environ["git"], "sleep_study", "pre_process")
        self.nn = os.path.join(os.environ['git'], "sleep_study", "nn")
        pass
    
    def set(self):
        """
        この操作は一度しか行わない
        1. クラスのプロパティを全て追加する（ただし callble の時のみ） \n
        2. tensorflow, wandb のログを表示しない
        """
        if not SetsPath.executedFlag:
            for _, value in self.__dict__.items():
                if callable(value) == False:
                    sys.path.append(value)
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            os.environ["WANDB_SILENT"] = "true"
            SetsPath.executedFlag = True

# ================================================ #
# *ディレクトリとプロジェクト名を関連付けるクラス
# ================================================ #
  
class FindsDir(object):
    """target_problem によって保存するフォルダを分けるためのクラス

    Args:
        object ([type]): [description]
    """
    def __init__(self, target_problem):
        self.dirDict = {"sleep" : "sleep_study",
                        "test" : "test",
                        "oxford" : "oxford"}
        self.target_problem = target_problem
        
    def returnDirName(self):
        """ディレクトリ名だけが欲しい時はこちらのメソッドを呼び出す
        (ex)
        >> instance = FindsDir("sleep")
        >> instance.returnDirName()
        >> "sleep_study"
        Returns:
            [type]: [description]
        """
        dirName = self.dirDict[self.target_problem]
        return dirName
    
    def returnFilePath(self):
        """プロジェクトのパスが欲しい時はこちらのメソッドを呼び出す
        (ex)
        >> instance = FindsDir("sleep")
        >> instance.returnFilePath()
        Returns:
            [type]: [description]
        """
        path = os.environ['sleep']
        return path

    def returnDataPath(self):
        path = os.path.join(os.environ['sleep'],
                            "datas")
        return path
# ================================================ #
# *            試験用メイン関数
# ================================================ #

if __name__ == '__main__':
    m_setsPath = SetsPath()
    m_findsDir = FindsDir(target_problem='sleep')