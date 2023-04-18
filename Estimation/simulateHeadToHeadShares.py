import pandas as p
import numpy as np
from tqdm import tqdm, trange
from buildPlots import getFullData, constructSpecDict
from collections import OrderedDict
from getPythonModels import loadPooledModels

def constructFullSpecDicts(type='Car', relRange = 300, truck=False):
    fullData = getFullData()[type]
    groupNames = list(set(fullData.loc[:, 'groupName']))
    r = OrderedDict()
    newR = OrderedDict()
    for groupName in groupNames:
        r[groupName] = OrderedDict()
        subData = fullData.loc[fullData.loc[:, 'groupName']==groupName, :]
        for ind, row in subData.iterrows():
            r[groupName][row['model']] = constructSpecDict(row, truck=truck)
        newR[groupName] = p.DataFrame.from_dict(r[groupName])
        newR[groupName] = newR[groupName].transpose()
        newR[groupName].loc[:, 'bevRange'] = newR[groupName].loc[:, 'bevRange']-relRange
        newR[groupName].loc[:, 'price'] = newR[groupName].loc[:, 'price']/1000
        newR[groupName] = newR[groupName].transpose()

    return newR

def simulateParams(modelInfo, nSim=100):

    mean = modelInfo['coef']
    if ('bestModel' in modelInfo.keys()):
        cov = modelInfo['vc']
    else:
        cov = modelInfo['robustVC']
    # cov = modelInfo['robustVC']

    samples = np.random.multivariate_normal(mean, cov, nSim)
    samples = p.DataFrame(samples, columns=modelInfo['params'])
    return samples

def getMeanCols(paramDF):
    r = []
    for col in paramDF.columns:
        if('SD' not in col):
            r.append(col)

    return r


def getSDCols(paramDF):
    r = []
    for col in paramDF.columns:
        if ('SD' in col):
            r.append(col)

    return r

def getSDColFromMean(meanCol):
    if('Mean' not in meanCol):
        return None
    else:
        return meanCol[:meanCol.index('Mean')]+'SD'

def simulateIndParams(row, meanCols, nInd):
    mean = []
    cov = []
    for meanCol in meanCols:
        mean.append(row[meanCol])
        if(getSDColFromMean(meanCol)==None):
            cov.append(0)
        else:
            cov.append(row[getSDColFromMean(meanCol)]*row[getSDColFromMean(meanCol)])


    cov = np.diag(cov)
    indParams = np.random.multivariate_normal(mean, cov, nInd)
    cols = []

    for meanCol in meanCols:
        if ('Mean' in meanCol):
            cols.append(meanCol[:meanCol.index('Mean')])
        else:
            cols.append(meanCol)
    indParams = p.DataFrame(indParams, columns=cols)
    return indParams


def constructPrefCoefs(modelInfo, nSim=100, nInd=1):
    if(modelInfo['modelType']=='mnl'):
        nInd = 1
    paramDF = simulateParams(modelInfo, nSim)
    indParams = []
    meanCols = getMeanCols(paramDF)
    for ind, row in tqdm(paramDF.iterrows(), disable=False):
        tempIndParams = simulateIndParams(row, meanCols, nInd)
        indParams.append(tempIndParams)



    return indParams

def calculateMeanChoiceProbabilites(modelsDF, paramsDF):

    colsToKeep = []
    if('phevRange' in modelsDF.index):
        modelsDF.loc['phev20', :] = (40-modelsDF.loc['phevRange', :])/20
        modelsDF.loc['phev40', :] = (modelsDF.loc['phevRange', :]-20)/20
        modelsDF = modelsDF.drop('phevRange')
    for tempInd in modelsDF.index:
        if(tempInd=='phevRange'):
            colsToKeep.append('phev20')
            colsToKeep.append('phev40')
        else:
            colsToKeep.append(tempInd)
    paramsDF = paramsDF.loc[:, colsToKeep]

    for col in paramsDF.columns:
        # if()
        if(col != 'price'):
            paramsDF.loc[:, col] = paramsDF.loc[:, col]*paramsDF.loc[:, 'price']

    paramsDF.loc[:, 'price'] = -paramsDF.loc[:, 'price']
    modelsDF = modelsDF.fillna(0)
    utilities = np.array(paramsDF)@np.array(modelsDF)


    utilities = p.DataFrame(utilities, columns=modelsDF.columns)



    utilities.loc[:, 'evExpUtility'] = np.exp(utilities.loc[:, list(utilities.columns)[0]])
    utilities.loc[:, 'cvExpUtility'] = np.exp(utilities.loc[:, list(utilities.columns)[1]])
    utilities.loc[:, 'evShare'] = utilities.loc[:, 'evExpUtility']/(utilities.loc[:, 'evExpUtility']+utilities.loc[:, 'cvExpUtility'])
    return np.mean(utilities.loc[:, 'evShare'])


def calculateMultipleMeanChoiceProbabilities(modelInfo, headToHeadData, nSim, nInd):
    r = OrderedDict()
    params = constructPrefCoefs(modelInfo, nSim, nInd)
    for name, df in headToHeadData.items():
        r[name] = []
        for paramSample in tqdm(params, desc='Working on {}'.format(name)):
            r[name].append(calculateMeanChoiceProbabilites(df, paramSample))
    r = p.DataFrame.from_dict(r)
    return r

def simByType(type='Car', nSim=20000, nInd=20000):
    models = loadPooledModels(['2018', '-lin', '-{}'.format(type.lower())], antiDisc=['base', 'demo'])
    # file = files[0]
    if(type in ['truck', 'Truck']):
        truck = True
    else:
        truck = False
    headToHeadData = constructFullSpecDicts(type, truck=truck)
    print(headToHeadData)
    for name, modelInfo in models.items():
        r = calculateMultipleMeanChoiceProbabilities(modelInfo, headToHeadData, nSim=nSim, nInd=nInd)
        saveFile = 'HeadToHeadSims/{}.csv'.format(name)
        print(saveFile)
        r.to_csv(saveFile, index=False)
    return r

if __name__=='__main__':
    np.random.seed(2241995)
    rCar = simByType('Car')
    rSUV = simByType('SUV')