import sys, wandb, tensorboard

class MakeTrainTestData(object):
    def __init__(self,
                 isWandb = False,
                 isTensorBoard = False,
                 architecture = "fnn"):
        self.isWandb = isWandb
        self.isTensorBoard = isTensorBoard
        self.architecture = architecture
        pass
    
    def initWandb(self):
        """wandb でログを取る
        """
        pass
    
    def initTensorBoard(self):
        """tensorboard でログを取る
        """
        pass
    
    def make4Fnn(self):
        """fnn 用の前処理
        """
    
    def make4Cnn_1d(self):
        """1d-CNN の前処理
        """
        
    def make4Cnn_2d(self):
        """2d-CNN の前処理
        """
        pass
    
    
    def make(self):
        """make メソッドの中でネットワーク構造で場合分けを行い前処理を変える
        """
        # tensorboard や wandb を使う場合はここで呼ばれる
        if self.isWandb:
            self.initWandb()
            
        if self.isTensorBoard:
            self.initTensorBoard()
        
        # 前処理を architecture の名前によって判断
        if self.architecture == "fnn":
            return self.make4Fnn()
            
        if self.architecture == "1d_cnn":
            return self.make4Cnn_1d

        if self.architecture == "2d_cnn":
            return self.make4Cnn_2d()
        
        else:
            print(f"There is no {self.architecture} method in MakeTrainTestData class")
            sys.exit(1)