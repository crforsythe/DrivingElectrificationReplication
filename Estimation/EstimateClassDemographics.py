import pandas as p
import numpy as np
from tqdm import tqdm, trange
from collections import OrderedDict
from OutputLCLTables import getClassShare, loadAllData
import stata_setup as s
from copy import deepcopy
try:
    s.config('/Applications/Stata', 'mp')
except:
    s.config("C:\Program Files\Stata17", 'mp')
from pystata import stata
def loadProbabilities(numClasses):
    file = 'Data/UpdatedClassProbabilities.xlsx'
    sheetName = 'classCount-{}'.format(int(numClasses))
    try:
        data = p.read_excel(file, sheetName)
        return data
    except:
        print(sheetName)
        return


def loadDemographics():
    try:
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Pooled/PooledWeighted.xlsx'
        data = p.read_excel(file, 'demo-truck')
        return data
    except:
        print('Windows Machine')

def renameClassProbColumns(classProbabilities, numClasses):
    renameDict = {}
    for i in range(numClasses):
        renameDict[i+1] = 'Class{}Probability'.format(i+1)
    classProbabilities = classProbabilities.rename(renameDict, axis=1)
    return classProbabilities

def predictClass(classProbabilities, numClasses):
    newCol = []
    for ind, row in tqdm(classProbabilities.iterrows(), disable=True):
        bestClass = 0
        bestProb = 0
        for i in range(numClasses):
            tempClass = i+1
            tempProb = row[tempClass]

            if(tempProb>bestProb):
                bestClass = tempClass
                bestProb = tempProb
        newCol.append(bestClass)

    classProbabilities.loc[:, 'ClassPrediction'] = newCol
    return classProbabilities


def loadAllByNumClasses(numClasses):
    classProbabilities = loadProbabilities(numClasses)
    classProbabilities = predictClass(classProbabilities, numClasses)
    classProbabilities = renameClassProbColumns(classProbabilities, numClasses)
    demographics = loadDemographics()
    merged = classProbabilities.merge(demographics, on='ID')

    r = OrderedDict()
    r['Full-{}'.format(numClasses)] = merged
    for i in range(numClasses):
        r['{}-{}'.format(i+1, numClasses)] = merged.loc[merged.loc[:, 'ClassPrediction']==i+1, :]

    return r

def loadAll(save=True):
    r = OrderedDict()
    for i in trange(2,9):
        try:
            r[i] = loadAllByNumClasses(i)
        except:
            print(i)


    if(save):
        xlw = p.ExcelWriter('Data/ClassAssignments.xlsx')
        for numClasses, demogDict in tqdm(r.items()):
            for sheetName, demogDF in demogDict.items():
                demogDF.to_excel(xlw, sheetName, index=False)
        xlw.close()

    return r

def loadMarginalDamages(vehType=None):
    file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Pooled/MarginalDamages.csv'
    try:
        data = p.read_csv(file)
    except:
        print('Windows machine')

    if(vehType=='None'):
        return data
    else:
        return data.loc[data.loc[:, 'vehType']==vehType, :]


def outputDemogDifferences(numClasses=5, vars = ('envLifeRatings_r3', 'envLifeRatings_r4', 'NumVehiclesOwned', 'Clean Age', 'Clean Income', 'Woman', 'AP2eGridDamages', 'PopulationDensity', 'co2'), returnTable = True):
    data = p.read_excel('Data/ClassAssignments.xlsx', 'Full-{}'.format(numClasses))
    data.loc[:, 'Woman'] = 0
    data.loc[data.loc[:, 'Clean gender']=='Woman', 'Woman'] = 1
    data = data.merge(loadMarginalDamages('truck'), on='ID')
    varsToKeep = ['ClassPrediction']

    varsToKeep.extend(vars)
    data = data.loc[:, varsToKeep]

    groupedData = data.groupby('ClassPrediction')
    counts = groupedData.count()
    means = groupedData.mean()
    std = groupedData.std()
    se = std/np.sqrt(counts)

    means = means.transpose()
    se = se.transpose()
    t = loadAllData()
    maxIndex = list(getClassShare(t, numClasses, string=False, sort=True).keys())[0]

    classNames = ['\makecell{Tech-\\\\Indifferent}' ,  '\makecell{Tech-\\\\Skeptical}',  'Catch-All' , '\makecell{Operating-Cost\\\\Sensitive}' , '\makecell{Tech-\\\\Enthusiast}']
    varNames = ['\\threat', '\human', '\\veh', '\\age', '\inc', '\woman', '\damages', '\popDen', '\co']

    varMapping = dict(zip(varNames, vars))
    reverseClassNameDict = dict(zip(classNames, range(1, len(classNames)+1)))

    classNameDict = dict(zip(range(1, len(classNames)+1), classNames))

    # means = means.rename(dict(zip(list(means.columns), classNames)), axis=1)
    means = means.rename(dict(zip(list(means.index), varNames)), axis=0)

    # se = se.rename(dict(zip(list(se.columns), classNames)), axis=1)
    se = se.rename(dict(zip(list(se.index), varNames)), axis=0)

    r = OrderedDict()
    percCols = ['Woman']
    for ind in means.index:
        r[ind] = OrderedDict()
        for col in means.columns:
            # rawIndex = reverseClassNameDict[col]
            rawIndex = col
            rawVar = varMapping[ind]
            sig = False
            positive = False
            if(rawIndex!=maxIndex):
                print('ind-{}; col-{}'.format(rawVar, rawIndex))
                tTestInfo = performTTest(data, [maxIndex, rawIndex], rawVar)
                print(tTestInfo)
                print('-'*100)
                if(tTestInfo['r(p)']<0.05):
                    sig = True

                if(tTestInfo['r(t)']>0):
                    positive = True

                seVal = tTestInfo['r(sd_2)']/np.sqrt(tTestInfo['r(N_2)'])
                meanVal = tTestInfo['r(mu_2)']
            else:
                tTestInfo = performTTest(data, [maxIndex], rawVar)
                seVal = tTestInfo['r(se)']
                meanVal = tTestInfo['r(mu_1)']
            if(ind in percCols or 'Woman' in ind):
                r[ind][col] = '\makecell{' + '{:0.0f}\%\\\\({:.1f}\%)'.format(meanVal*100, seVal*100) + '}'
            elif(ind == 'PopulationDensity' or 'Density' in ind):
                r[ind][col] = '\makecell{' + '{:0.0f}\\\\({:.0f})'.format(meanVal, seVal) + '}'
            else:
                r[ind][col] = '\makecell{' + '{:0.1f}\\\\({:.2f})'.format(meanVal, seVal) + '}'

            if (sig):
                if (positive):
                    colorEntry = '\\textcolor{red}{'
                else:
                    colorEntry = '\\textcolor{blue}{'

                r[ind][col] = colorEntry + r[ind][col] + '}'
    r = p.DataFrame.from_dict(r)
    r = r.transpose()

    print(r)

    sortedIndices = getClassShare(t, numClasses, string=False, sort=True).keys()

    # r = r.loc[:, ]

    sortedNames = []

    for sortedIndex in sortedIndices:
        sortedNames.append(classNameDict[sortedIndex])

    # r = r.loc[:, sortedNames]
    r.to_latex('TableLCL/demo-LC-{}.tex'.format(numClasses), index=True, escape=False, index_names='')
    if(returnTable):
        return r
    else:
        return data

def performTTest(data, classes, var):
    stata.run('clear')
    if(' ' in var):
        var = var.replace(' ', '')
    dataCopy = deepcopy(data)
    dataCopy = dataCopy.loc[dataCopy.loc[:, 'ClassPrediction'].isin(classes), :]
    dataCopy.loc[:, 'ClassPrediction'] = dataCopy.loc[:, 'ClassPrediction'].replace(dict(zip(classes, range(len(classes)))))
    dataCopy = dataCopy.sort_values('ClassPrediction')
    # return dataCopy
    stata.pdataframe_to_data(dataCopy)
    print(len(classes))
    if(len(classes)>1):
        stata.run('ttest {}, by(ClassPrediction) unequal'.format(var))
    else:
        print('lol')
        stata.run('ttest {}==0'.format(var))

    return stata.get_return()



# a = loadAll()
data = p.read_excel('Data/ClassAssignments.xlsx', 'Full-{}'.format(5))
t = outputDemogDifferences(returnTable=True)
print(t)
m = loadMarginalDamages('truck')

j = data.merge(m, on='ID', how='inner')

# r = performTTest(j, [1,2], 'PopulationDensity')
r2 = performTTest(j, [1,2], 'Clean Age')
r3 = performTTest(j, [1], 'Clean Age')