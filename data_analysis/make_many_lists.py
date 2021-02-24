import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import array, append, average, median, std
from tqdm import tqdm


class Person():

    def __init__(self, name, dim = 512):
        self.name = name
        self.nr1 = []
        self.nr2 = []
        self.nr34 = []
        self.rem = []
        self.wake = []
        self.nr4LabelFrom = 0
        self.nr3LabelFrom = 1
        self.nr2LabelFrom = 2
        self.nr1LabelFrom = 3
        self.rLabelFrom = 4
        self.wLabelFrom = 5
        self.wLabelTo = 0
        self.rLabelTo = 1
        self.nr1LabelTo = 2
        self.nr2LabelTo = 3
        self.nr34LabelTo = 4
        self.dim = dim
        self.spectrumMatrix = array([])
        self.spectrumAverageNr1 = array([])
        self.spectrumMedianNr1 = array([])
        self.spectrumStdNr1 = array([])
        self.spectrumAverageNr2 = array([])
        self.spectrumMedianNr2 = array([])
        self.spectrumStdNr2 = array([])
        self.spectrumAverageNr34 = array([])
        self.spectrumMedianNr34 = array([])
        self.spectrumStdNr34 = array([])
        self.spectrumAverageRem = array([])
        self.spectrumMedianRem = array([])
        self.spectrumStdRem = array([])
        self.spectrumAverageWake = array([])
        self.spectrumMedianWake = array([])
        self.spectrumStdWake = array([])
        pass
    
    def changeLabel(self, record):
        if record.PSG == self.nr4LabelFrom or record.PSG == self.nr3LabelFrom:
            record.PSG = self.nr34LabelTo
            self.nr34.append(record)
            
        elif record.PSG == self.nr2LabelFrom:
            record.PSG = self.nr2LabelTo
            self.nr2.append(record)
            
        elif record.PSG == self.nr1LabelFrom:
            record.PSG = self.nr1LabelTo
            self.nr1.append(record)
            
        elif record.PSG == self.rLabelFrom:
            record.PSG = self.rLabelTo
            self.rem.append(record)
            
        elif record.PSG == self.wLabelFrom:
            record.PSG = self.wLabelTo
            self.wake.append(record)

        else:
            print(f"unknown sleep stage {record.PSG} came !!")
            sys.exit(1)
        return
    
    def changeLabelAll(self, recordList):
        for record in recordList:
            self.changeLabel(record)
        return
    
    def expandSpectrum(self, record):
        return record.spectrum

    def makeMatrix(self, recordList, spectrumMatrix):
        for record in tqdm(recordList):
            spectrumMatrix = append(spectrumMatrix, self.expandSpectrum(record))
        spectrumMatrix = spectrumMatrix.reshape(-1, self.dim)
        return spectrumMatrix
    
    def calculate(self, spectrumMatrix):
        spectrumAverage = average(spectrumMatrix, axis = 0)
        spectrumMedian = median(spectrumMatrix, axis = 0)
        spectrumStd = std(spectrumMatrix, axis = 0)
        return spectrumAverage, spectrumMedian, spectrumStd
    
    def makeGraph(self, 
                  spectrumAverage, 
                  spectrumMedian, 
                  spectrumStd, 
                  filePath, 
                  startHz = 0, 
                  endHz = 8, 
                  title = 'tmp'):
        x = np.linspace(startHz, endHz, self.dim)
        sns.set()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(x, spectrumAverage, label = 'average')
        # 中央値は平均値と大差がなさそうなので，とりあえずコメントにしておく
        # ax1.plot(self.spectrumMedian, label = 'median')
        ax1.fill_between(x, spectrumAverage, 
                         spectrumAverage + spectrumStd, 
                         alpha = 0.2, 
                         facecolor = 'blue', 
                         label = 'upper sigma')
        ax1.fill_between(x, spectrumAverage, 
                         spectrumAverage - spectrumStd, 
                         alpha = 0.2, 
                         facecolor = 'blue', 
                         label = 'lower sigma')
        plt.title(title, fontsize = 20)
        plt.xlabel('Freq [Hz]', fontsize = 20)
        plt.ylabel('dB [-]', fontsize = 20)
        plt.legend(fontsize = 20)
        plt.tight_layout()
        if filePath == None:
            pass
        else:
            plt.savefig(filePath)
    
    def existingFileChecker(self, 
                            sleepStage = "tmp"):
        if os.path.exists('figure') != True:
            os.mkdir('figure')
        filePath = os.path.join(os.getcwd(), 'figure', sleepStage+'.png')
        counter = 0
        while (os.path.exists(filePath)):
            filePath = os.path.join(os.getcwd(), 'figure', sleepStage+'_'+str(counter)+'.png')
            counter += 1
        return filePath
             