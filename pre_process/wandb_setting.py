# ================================================ #
# *            ライブラリのインポート
# ================================================ #

import os, sys, wandb, datetime

# ================================================ #
# *       wandb のメソッドをまとめたクラス
# ================================================ #

class WandbSettings():
    
    def __init__(self, name, project, silent = True):
        self.name = name
        self.project = project
        self.id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if silent:
            os.environ["WANDB_SILENT"] = "true"
    
    def myInit(self, name, project, job_type, **kwargs):
        # NOTE : このままだと id 以外使う意味ない
        run = wandb.init(name = f"{name} : {self.id}", 
                         project = project,
                         job_type = job_type,
                        config = self.setConfig(kwargs))
        return run
    
    
    
