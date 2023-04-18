import pandas as p
import numpy as np
from tqdm import tqdm

def getPHEVFuelShare(range):
    data = getCleanPHEVFuelShareData()


    minSqDiff = np.inf
    rProp = np.inf

    for dRange, dProp in data.items():
        tempSqDiff = np.power(dRange-range, 2)
        if(tempSqDiff<minSqDiff):
            minSqDiff = tempSqDiff
            rProp = dProp

    return rProp

def getCleanPHEVFuelShareData():
    file = 'Data/PHEV Utilization Data/Multi-DayUF.xlsx'
    data = p.read_excel(file)

    renameDict = {'Distance (mi)':'Distance (mi).0', 'UF':'UF.0'}
    data = data.rename(renameDict, axis=1)
    r = {}
    for i in range(3):
        tempRangeCol = 'Distance (mi).{}'.format(i)
        tempShareCol = 'UF.{}'.format(i)

        for ind, row in data.iterrows():
            r[row[tempRangeCol]] = row[tempShareCol]


    return r

if __name__=='__main__':
    t = getCleanPHEVFuelShareData()

