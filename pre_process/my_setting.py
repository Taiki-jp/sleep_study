import os
import sys


class SetsPath(object):
    # 一度しか set メソッドを実行しないためのクラス変数
    executed_flag = False

    def __init__(self):
        self.data_analysis = os.path.join(
            os.environ["git"], "sleep_study", "data_analysis"
        )
        self.pre_process = os.path.join(
            os.environ["git"], "sleep_study", "pre_process"
        )
        self.nn = os.path.join(os.environ["git"], "sleep_study", "nn")
        pass

    def set(self):
        if not SetsPath.executed_flag:
            for _, value in self.__dict__.items():
                if callable(value) is False:
                    sys.path.append(value)
            # NOTE : どこで初期化しても設定は保持される（出来るだけプログラムの最初の方に設定したい）
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            os.environ["WANDB_SILENT"] = "true"
            SetsPath.executed_flag = True


# ================================================ #
# *ディレクトリとプロジェクト名を関連付けるクラス
# ================================================ #


class FindsDir(object):
    """target_problem によって保存するフォルダを分けるためのクラス

    Args:
        object ([type]): [description]
    """

    def __init__(self, target_problem):
        self.dir_dict = {
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
        dir_name = self.dir_dict[self.target_problem]
        return dir_name

    def returnFilePath(self):
        """プロジェクトのパスが欲しい時はこちらのメソッドを呼び出す
        (ex)
        >> instance = FindsDir("sleep")
        >> instance.returnFilePath()
        Returns:
            [type]: [description]
        """
        path = os.environ["sleep"]
        return path

    def returnDataPath(self):
        path = os.path.join(os.environ["sleep"], "datas")
        return path


# ================================================ #
# *            試験用メイン関数
# ================================================ #
class SetsPathUnderProject:
    pass


# ================================================ #
# *            試験用メイン関数
# ================================================ #

if __name__ == "__main__":
    sets_path = SetsPath()
    finds_dir = FindsDir(target_problem="sleep")
