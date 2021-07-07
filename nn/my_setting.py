import os
import sys


class SetsPath(object):
    # 一度しか set メソッドを実行しないためのクラス変数
    executedFlag = False

    def __init__(self):
        self.dataAnalysis = os.path.join(
            os.environ["git"], "sleep_study", "data_analysis"
        )
        self.preProcess = os.path.join(
            os.environ["git"], "sleep_study", "pre_process"
        )
        self.nn = os.path.join(os.environ["git"], "sleep_study", "nn")
        pass

    def set(self):
        """
        この操作は一度しか行わない
        1. クラスのプロパティを全て追加する（ただし callble の時のみ） \n
        2. tensorflow, wandb のログを表示しない
        """
        if not SetsPath.executedFlag:
            for _, value in self.__dict__.items():
                if callable(value) is False:
                    sys.path.append(value)
            # ANCHOR : ここで実行した設定を呼び出し元のファイルで継続するにはどうするとよいか？
            # TODO : 方法１（別々のosを呼び出してprintで値を確認する）
            # TODO : 方法２（ここで用いたosを呼び出してprintする）
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            os.environ["WANDB_SILENT"] = "true"
            # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            SetsPath.executedFlag = True


class FindsDir(object):
    """target_problem によって保存するフォルダを分けるためのクラス"""

    def __init__(self, target_problem):
        """初期化メソッド

        Args:
            target_problem ([string]): [dirDictの中からtarget_problemを選ぶ]
        """
        self.dirDict = {
            "sleep": "sleep_study",
            "test": "test",
            "oxford": "oxford",
        }
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
        path = os.path.join(os.environ["sleep"])
        return path
