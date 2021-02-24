import sys
import os
sys.path.append('pythonscript')
sys.path.append('makeGraph')
import numpy as np
from make_many_lists import Person
import file_reader as reader
from tqdm import tqdm
from make_spectrum_graph import MakeSpectrumGraph

recordMatrix = reader.loadNormal()

patientNameList = \
[
    "H_Li", 
    "H_Murakami", 
    "H_Yamamoto",  
    "H_Kumazawa",
    "H_Hayashi", 
    "H_Kumazawa_F", 
    "H_Takadama", 
    "H_Hiromoto", 
    "H_Kashiwazaki"
]

personDict = {patientNameList[i] : Person(patientNameList[i]) for i in range(len(patientNameList))}

for counter, person in enumerate(personDict):
    personDict[person].changeLabelAll(recordMatrix[counter])
    tmp = np.array([])
    tmp = personDict[person].makeMatrix(personDict[person].nr1, tmp)
    personDict[person].spectrumAverageNr1, personDict[person].spectrumMedianNr1, personDict[person].spectrumStdNr1 = personDict[person].calculate(spectrumMatrix = tmp)
    # tmp = np.array([])
    # tmp = personDict[person].makeMatrix(personDict[person].nr2, tmp)
    # personDict[person].spectrumAverageNr2, personDict[person].spectrumMedianNr2, personDict[person].spectrumStdNr2 = personDict[person].calculate(spectrumMatrix = tmp)
    tmp = np.array([])
    tmp = personDict[person].makeMatrix(personDict[person].nr34, tmp)
    personDict[person].spectrumAverageNr34, personDict[person].spectrumMedianNr34, personDict[person].spectrumStdNr34 = personDict[person].calculate(spectrumMatrix = tmp)    
    tmp = np.array([])
    tmp = personDict[person].makeMatrix(personDict[person].rem, tmp)
    personDict[person].spectrumAverageRem, personDict[person].spectrumMedianRem, personDict[person].spectrumStdRem = personDict[person].calculate(spectrumMatrix = tmp)    
    tmp = np.array([])
    tmp = personDict[person].makeMatrix(personDict[person].wake, tmp)
    personDict[person].spectrumAverageWake, personDict[person].spectrumMedianWake, personDict[person].spectrumStdWake = personDict[person].calculate(spectrumMatrix = tmp)

    # 画像を保存したい時は false にする
    if False:
        filePath = personDict[person].existingFileChecker(f"nr1_{person}")
        personDict[person].makeGraph(personDict[person].spectrumAverageNr1,
                                     personDict[person].spectrumMedianNr1, 
                                     personDict[person].spectrumStdNr1,
                                     filePath = filePath,
                                     title = "NR1")

        #filePath = personDict[person].existingFileChecker("nr2")
        #personDict[person].makeGraph(personDict[person].spectrumAverageNr2,
        #                             personDict[person].spectrumMedianNr2, 
        #                             personDict[person].spectrumStdNr2,
        #                             filePath = filePath,
        #                             title = "NR2")    

        filePath = personDict[person].existingFileChecker(f"nr34_{person}")
        personDict[person].makeGraph(personDict[person].spectrumAverageNr34,
                                     personDict[person].spectrumMedianNr34, 
                                     personDict[person].spectrumStdNr34,
                                     filePath = filePath,
                                     title = "NR34")

        filePath = personDict[person].existingFileChecker(f"rem_{person}")
        personDict[person].makeGraph(personDict[person].spectrumAverageRem,
                                     personDict[person].spectrumMedianRem, 
                                     personDict[person].spectrumStdRem,
                                     filePath = filePath,
                                     title = "REM")

        filePath = personDict[person].existingFileChecker(f"wake_{person}")
        personDict[person].makeGraph(personDict[person].spectrumAverageWake,
                                     personDict[person].spectrumMedianWake, 
                                     personDict[person].spectrumStdWake,
                                     filePath = filePath,
                                     title = "WAKE")
    
# とりあえず，各自の同じ睡眠段階の内積を計算する？
# その次に，異なる睡眠段階についても内積を計算する？
# 他にベクトルの近さの指標になりそうなものを探す？
# [MEMO : np.linalg.norm(array, cord = 2) で L2 ノルムが計算できる]