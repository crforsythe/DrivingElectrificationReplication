import pandas as p
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
import pickle
from collections import OrderedDict
from tqdm import tqdm
from copy import deepcopy
from getPythonModels import loadPooledModels


def loadMarketData(type = 'Car', newFuelPrices = False):
    file = 'Data/2022 CAFE FRIA/Central Analysis/input/market_data_ref.xlsx'

    data = p.read_excel(file, 'Vehicles', skiprows=1)
    data.loc[:, 'price'] = data.loc[:, 'MSRP']/1000
    data = data.loc[data.loc[:, 'Sales']>0, :]
    data = data.loc[data.loc[:, 'Technology Class'].str.contains(type), :]
    data.loc[:, 'marketShare'] = data.loc[:, 'Sales']/np.sum(data.loc[:, 'Sales'])
    data.loc[:, 'type'] = 'CV'
    data.loc[data.loc[:, 'Fuel Share (E)']==1, 'type'] = 'BEV'
    data.loc[(data.loc[:, 'Fuel Share (E)']<1) & (data.loc[:, 'Fuel Share (G)']<1), 'type'] = 'PHEV'

    hybridCols = ['P2HCR0', 'P2HCR1', 'P2HCR2', 'SHEVP2', 'SHEVPS']
    for hybridCol in hybridCols:
        data.loc[data.loc[:, hybridCol]=='USED', 'type'] = 'HEV'

    data = data.sort_values('type')

    data.loc[:, 'horseWeightRatio'] = data.loc[:, 'Vehicle Power']/data.loc[:, 'Curb Weight']
    data.loc[:, 'acc'] = np.power(data.loc[:, 'horseWeightRatio']/0.3452, -1/0.88)
    # data.loc[:, 'accNEMS'] = np.power(data.loc[:, 'horseWeightRatio']*np.exp(-0.00275), -0.776)

    if(newFuelPrices):
        gasPrice = 427.1
        elecCost = 13.83
    else:
        gasPrice = 263.6
        elecCost = 13.04

    data.loc[:, 'oc'] = gasPrice/data.loc[:, 'Fuel Economy (G)']
    data.loc[data.loc[:, 'type']=='BEV', 'oc'] = (33.705*elecCost)/(data.loc[:, 'Fuel Economy (E)'])
    data.loc[:, 'meanEconomy'] = 1/((1/(data.loc[:, 'Fuel Share (G)']*data.loc[:, 'Fuel Economy (G)']))+(1/(data.loc[:, 'Fuel Share (E)']*data.loc[:, 'Fuel Economy (E)'])))

    data.loc[:, 'hev'] = 0
    data.loc[:, 'bev'] = 0
    data.loc[:, 'bevRange'] = 0

    data.loc[data.loc[:, 'type']=='HEV', 'hev'] = 1
    data.loc[data.loc[:, 'type']=='BEV', 'bev'] = 1
    data.loc[:, 'bevRange'] = 0
    data.loc[:, 'bevFC'] = 0
    data = data.loc[data.loc[:, 'oc']==data.loc[:, 'oc'], :]




    return data

def estimateMeanMarketShare(marketData, ASCs=None, utilities=None, output=False):
    if(type(ASCs)==type(None)):
        ASCs = np.array([0]*len(marketData))
    if (type(utilities) == type(None)):
        utilities = np.array([0] * len(marketData))

    if(utilities.size==len(marketData)):
        utilities = ASCs+utilities
        expUtilities = np.exp(utilities)
        probabilities = expUtilities/np.sum(expUtilities)
    else:
        ASCs = ASCs.reshape((-1,1))

        utilities = utilities+ASCs
        expUtilities = np.exp(utilities)
        sumExpUtilities = np.sum(expUtilities, axis=0).reshape((1,-1))


        probabilities = expUtilities / sumExpUtilities

        probabilities = np.mean(probabilities, axis=1)

    if(output):
        print('ASC Shape: {}'.format(ASCs.shape))
        print('Utilities Shape: {}'.format(utilities.shape))

    return probabilities

def loadModels():
    with open('models/models.dat', 'rb') as f:
        r = pickle.load(f)

        f.close()

    return r

def loadPythonModels(setOC = 0):
    r = OrderedDict()

    r['car'] = OrderedDict()
    r['suv'] = OrderedDict()

    carModel = loadPooledModels(['2018', 'car', 'lin', 'sim'])
    suvModel = loadPooledModels(['2018', 'suv', 'lin', 'sim'])

    r['car']['simple'] = carModel[list(carModel.keys())[0]]
    r['suv']['simple'] = suvModel[list(suvModel.keys())[0]]

    carModel = loadPooledModels(['2018', 'car', 'lin', 'mixe'])
    suvModel = loadPooledModels(['2018', 'suv', 'lin', 'mixe'])

    r['car']['mixed'] = carModel[list(carModel.keys())[0]]
    r['suv']['mixed'] = suvModel[list(suvModel.keys())[0]]

    return r

def calculateKLDivergence(marketData, ASCs=None, utilities=None):
    if (type(ASCs) == type(None)):
        ASCs = np.array([0] * len(marketData))
    if (type(utilities) == type(None)):
        utilities = np.array([0] * len(marketData))

    baseShare = np.array(marketData.loc[:, 'marketShare'])
    predShare = estimateMeanMarketShare(marketData, ASCs, utilities=utilities)
    logDivTerm = np.log(predShare/baseShare)
    kl = predShare @ logDivTerm

    return kl

def calculateSquaredError(marketData, ASCs=None, utilities=None):
    if (type(ASCs) == type(None)):
        ASCs = np.array([0] * len(marketData))
    if (type(utilities) == type(None)):
        utilities = np.array([0] * len(marketData))

    baseShare = np.array(marketData.loc[:, 'marketShare'])*100
    predShare = estimateMeanMarketShare(marketData, ASCs, utilities=utilities)*100
    sqError = np.sum(np.power(baseShare-predShare, 2))
    return sqError

def estimateASCs(marketData, ASCs=None, utilities=None):
    if (type(ASCs) == type(None)):
        ASCs = np.array([0] * len(marketData))
    if (type(utilities) == type(None)):
        utilities = np.array([0] * len(marketData))
    lb = np.array([-np.inf] * len(marketData))
    ub = np.array([np.inf] * len(marketData))

    lb[0] = 0
    ub[0] = 0
    A = np.diag([1]*len(marketData))
    objFun = lambda x: calculateKLDivergence(marketData, x, utilities)
    tol = 1e-8
    linCon = LinearConstraint(A, lb, ub)
    res = None
    res = minimize(objFun, ASCs)

    res.x = res.x-res.x[0]

    marketData.loc[:, 'ASC'] = res.x
    marketData.loc[:, 'obj'] = res.fun
    marketData.loc[:, 'success'] = res.success
    return marketData

def simulateParams(modelInfo, nSim=100):

    mean = modelInfo['coef']

    if ('bestModel' in modelInfo.keys()):
        cov = modelInfo['vc']
    else:
        cov = modelInfo['robustVC']

    samples = np.random.multivariate_normal(mean, cov, nSim)
    samples = p.DataFrame(samples, columns=modelInfo['params'])
    return samples

def getMeanCols(paramDF):
    r = []
    for col in paramDF.columns:
        if('SD' not in col):
            r.append(col)

    return r

def getMeanColsModelInfo(modelInfo):
    r = []
    for col in modelInfo['params']:
        if ('SD' not in col):
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

def simulateMeanIndParams(modelInfo, nInd=1):
    meanCols = getMeanColsModelInfo(modelInfo)

    if (modelInfo['modelType'] == 'mnl'):
        nInd = 1

    mean = []
    cov = []
    newCols = []
    for meanCol in meanCols:
        if(getSDColFromMean(meanCol)==None):
            cov.append(0)
            newCols.append(meanCol)
        else:
            newCols.append(meanCol[:meanCol.index('Mean')])
            sdIndex = modelInfo['params'].index(getSDColFromMean(meanCol))
            cov.append(np.power(modelInfo['coef'][sdIndex], 2))
        meanIndex = modelInfo['params'].index(meanCol)
        mean.append(modelInfo['coef'][meanIndex])

    cov = np.diag(cov)
    simParams = np.random.multivariate_normal(mean, cov, nInd)
    df = p.DataFrame(simParams, columns=newCols)

    for meanCol in newCols:
        if(meanCol!='price'):
            df.loc[:, meanCol] = df.loc[:, meanCol]*df.loc[:, 'price']
    df.loc[:, 'price'] = -df.loc[:, 'price']
    return df




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
    paramsDF = paramsDF.loc[:, modelsDF.index]

    for col in paramsDF.columns:
        if(col != 'price'):
            paramsDF.loc[:, col] = paramsDF.loc[:, col]*paramsDF.loc[:, 'price']

    paramsDF.loc[:, 'price'] = -paramsDF.loc[:, 'price']
    utilities = np.array(paramsDF)@np.array(modelsDF)


    utilities = p.DataFrame(utilities, columns=modelsDF.columns)

    utilities.loc[:, 'evExpUtility'] = np.exp(utilities.loc[:, list(utilities.columns)[0]])
    utilities.loc[:, 'cvExpUtility'] = np.exp(utilities.loc[:, list(utilities.columns)[1]])
    utilities.loc[:, 'evShare'] = utilities.loc[:, 'evExpUtility']/(utilities.loc[:, 'evExpUtility']+utilities.loc[:, 'cvExpUtility'])
    return np.mean(utilities.loc[:, 'evShare'])

def calculateUtility(marketData, paramDF):
    relCols = []

    for col in marketData.columns:
        if(col in paramDF.columns):
            relCols.append(col)

    subMarketData = marketData.loc[:, relCols]
    subParamDF = paramDF.loc[:, relCols]
    subParamDF = subParamDF.transpose()

    utilities = np.array(subMarketData)@np.array(subParamDF)


    if(utilities.size==len(marketData)):
        utilities = utilities.reshape((-1,))
    return utilities

def summarizeCalibratesASCs(out):
    for vehType, dfDict in out.items():
        print(vehType)
        for modelType, outDict in dfDict.items():
            df = outDict['marketData']
            print(modelType)
            print('MSE: {}'.format(np.mean(np.power(df.loc[:, 'diff'], 2))))
            print('True Share BEV: {}'.format(np.sum(df.loc[df.loc[:, 'type']=='BEV', 'marketShare'])))
            print('Estimated Share BEV: {}'.format(np.sum(df.loc[df.loc[:, 'type'] == 'BEV', 'predictedMeanShare'])))
            print('Average ASC Magnitude: {}'.format(np.mean(np.abs(df.loc[:, 'ASC']))))
            print('Obj. Value: {}'.format(np.mean(np.abs(df.loc[:, 'obj']))))
            print('Converged: {}'.format(list(set(df.loc[:, 'success']))))
            print('-'*100)

def calibrateASCs(nInd=10, startVals=None, suffix='', altCarOCValue=None, altSUVOCValue = None):


    r0 = loadPythonModels()
    out = OrderedDict()
    typeMapping = {'car':'Car', 'suv':'SUV'}
    for vehType, modelTypeDict in r0.items():
        print(vehType)
        cleanVehicleType = typeMapping[vehType]
        marketData = loadMarketData(cleanVehicleType)
        out[vehType] = OrderedDict()
        for modelType, modelDict in modelTypeDict.items():
            if(vehType=='car' and type(altCarOCValue) != type(None)):

                print(altCarOCValue)
                indPrefCoefs = simulateMeanIndParams(setModelOC(modelDict, altCarOCValue), nInd)
            elif(vehType=='suv' and type(altSUVOCValue) != type(None)):

                print(altSUVOCValue)
                indPrefCoefs = simulateMeanIndParams(setModelOC(modelDict, altSUVOCValue), nInd)
            else:

                indPrefCoefs = simulateMeanIndParams(modelDict, nInd)



            # indPrefCoefs = simulateMeanIndParams(modelDict, nInd)
            utilities = calculateUtility(marketData, indPrefCoefs)

            if(startVals==None):
                if(modelType=='simple'):
                    marketData = estimateASCs(marketData, utilities=utilities)
                else:
                    marketData = estimateASCs(marketData, utilities=utilities, ASCs=np.array(marketData.loc[:, 'ASC']))

            else:
                tempStartVal = np.array(startVals[vehType][modelType]['marketData'].loc[:, 'ASC'])
                marketData = estimateASCs(marketData, utilities=utilities, ASCs=tempStartVal)

            marketData.loc[:, 'predictedMeanShare'] = estimateMeanMarketShare(marketData, np.array(marketData.loc[:, 'ASC']), utilities=utilities, output=False)
            marketData.loc[:, 'diff'] = marketData.loc[:, 'predictedMeanShare']-marketData.loc[:, 'marketShare']
            out[vehType][modelType] = {'marketData':deepcopy(marketData), 'params':indPrefCoefs}


    summarizeCalibratesASCs(out)


    if(startVals==None):
        saveFile = 'ASC/ASC{}.dat'.format(suffix)
    else:
        saveFile = 'ASC/ASC{}-refined.dat'.format(suffix)

    with open(saveFile, 'wb') as f:
        pickle.dump(out, f)
        f.close()
    return out

def loadASCs(refined=False, suffix=''):


    if(not refined):
        saveFile = 'ASC/ASC{}.dat'.format(suffix)
    else:
        saveFile = 'ASC/ASC{}-refined.dat'.format(suffix)
    try:
        with open(saveFile, 'rb') as f:
            r = pickle.load(f)
            f.close()
    except FileNotFoundError:
        print('{} not found'.format(saveFile))
        r = None
    return r

def getPredictedBEVShare(marketData):
    subData = marketData.loc[marketData.loc[:, 'type']=='BEV', :]

    return np.sum(subData.loc[:, 'predictedMeanShare'])

def applyOCs(ascData, newFuelPrices=False):
    if (newFuelPrices):
        gasPriceNew = 427.1
        elecCostNew = 13.83

        gasPriceOld = 263.6
        elecCostOld = 13.04
    else:
        gasPriceNew = 263.6
        elecCostNew = 13.04

        gasPriceOld = 427.1
        elecCostOld = 13.83

    gasRatio = gasPriceNew/gasPriceOld
    elecRatio = elecCostNew/elecCostOld

    ascData = deepcopy(ascData)
    ascData.index = range(len(ascData))
    ascData.loc[:, 'ocOG'] = ascData.loc[:, 'oc']
    ascData.loc[:, 'oc'] = gasRatio * ascData.loc[:, 'ocOG']
    ascData.loc[ascData.loc[:, 'type'] == 'BEV', 'oc'] = elecRatio * (ascData.loc[ascData.loc[:, 'type'] == 'BEV', 'ocOG'])
    ascData.loc[:, 'meanEconomy'] = 1 / ((1 / (ascData.loc[ascData.loc[:, 'type'] == 'PHEV', 'Fuel Share (G)'] * gasRatio)) + (
                1 / (ascData.loc[ascData.loc[:, 'type'] == 'PHEV', 'Fuel Share (E)'] * elecRatio)))*ascData.loc[ascData.loc[:, 'type'] == 'PHEV', 'ocOG']

    return ascData

def predictNewVehicles(marketData, params, newRange=300, relRange=300, ocPropCV=.5, ocPropHEV=.8, pricePremium=1, newFuelPrices=False, accelerationProp=1):
    relTypes = ['CV', 'HEV']

    subMarketData = deepcopy(marketData.loc[marketData.loc[:, 'type'].isin(relTypes), :])
    marketData.loc[:, 'Original'] = 1
    subMarketData.loc[:, 'Original'] = 0
    subMarketData.loc[:, 'hev'] = 0
    subMarketData.loc[:, 'bev'] = 1
    subMarketData.loc[:, 'bevRange'] = newRange-relRange
    subMarketData.loc[subMarketData.loc[:, 'type']=='CV', 'oc'] = subMarketData.loc[:, 'oc']*ocPropCV
    subMarketData.loc[subMarketData.loc[:, 'type'] == 'HEV', 'oc'] = subMarketData.loc[:, 'oc'] * ocPropHEV
    subMarketData.loc[:, 'bevFC'] = 1
    subMarketData.loc[:, 'nobevFC'] = 0
    subMarketData.loc[:, 'type'] = 'BEV'
    subMarketData.loc[:, 'Model'] = subMarketData.loc[:, 'Model']+'-Hypothetical BEV'
    subMarketData.loc[:, 'price'] = pricePremium*subMarketData.loc[:, 'price']
    subMarketData.loc[:, 'acc'] = accelerationProp*subMarketData.loc[:, 'acc']

    newMarketData = deepcopy(p.concat([marketData, subMarketData]))

    newMarketData.loc[:, 'nobevFC'] = 0

    if(newFuelPrices):
        newMarketData = applyOCs(newMarketData, newFuelPrices)

    utilities = calculateUtility(newMarketData, params)
    newMarketData.loc[:, 'predictedMeanShare'] = estimateMeanMarketShare(newMarketData, np.array(newMarketData.loc[:, 'ASC']), utilities=utilities, output=True)

    print('Old Share: {}'.format(getPredictedBEVShare(marketData)))
    print('New Share: {}'.format(getPredictedBEVShare(newMarketData)))

    return newMarketData

def predictAllNewVehicles(loadedData, newRange=300, newPropPricePremium = 1, newFuelPrices=False, acclerationProp=1, altOCPropValues = False):
    r = OrderedDict()
    for vehType, modelDict in loadedData.items():
        r[vehType] = OrderedDict()

        if(vehType=='car'):
            ocPropCV = .4724
            ocPropHEV = .7397
        elif(vehType=='suv'):
            ocPropCV = .4467
            ocPropHEV = .7444
        elif (vehType == 'pickup'):
            ocPropCV = .4467
            ocPropHEV = .7444

        if(altOCPropValues):
            ocPropCV = 1
            ocPropHEV = 1

        for modelType, outDict in modelDict.items():
            print('{}-{}'.format(vehType, modelType))
            r[vehType][modelType] = predictNewVehicles(outDict['marketData'], outDict['params'], newRange=newRange, ocPropCV=ocPropCV, ocPropHEV=ocPropHEV, pricePremium=newPropPricePremium, newFuelPrices=newFuelPrices, accelerationProp=acclerationProp)
            print('-'*100)

    return r

def updateASCs(nInd = 1000, suffix='', altCarOCValue=None, altSUVOCValue=None):
    baseASCs = loadASCs(suffix=suffix)
    r = calibrateASCs(nInd, baseASCs, suffix=suffix, altCarOCValue=altCarOCValue, altSUVOCValue=altSUVOCValue)
    return r

def setModelOC(model, oc=0):

    meanInds = []
    sdInds = []

    r = deepcopy(model)
    for i in range(len(r['params'])):
        param = model['params'][i]
        if('oc' in param):
            if('SD' not in param):
                meanInds.append(i)
            else:
                sdInds.append(i)

    for meanInd in meanInds:
        r['coef'][meanInd] = oc
        r['vc'][meanInd, :] = 0
        r['vc'][:, meanInd] = 0

    for sdInd in sdInds:
        r['coef'][sdInd] = 0
        r['vc'][sdInd, :] = 0
        r['vc'][:, sdInd] = 0

    return r

if __name__=='__main__':
    np.random.seed(2241995)

    loadedData = calibrateASCs(1000)
    loadedDataLow = calibrateASCs(1000, suffix='-LowAltOC', altCarOCValue=-232/1000, altSUVOCValue=-250/1000)
    loadedDataHigh = calibrateASCs(1000, suffix='-HighAltOC', altCarOCValue=-1378/1000, altSUVOCValue=-1438/1000)

    loadedData = updateASCs(20000)
    loadedDataLow = updateASCs(20000, suffix='-LowAltOC', altCarOCValue=-232/1000, altSUVOCValue=-250/1000)
    loadedDataHigh = updateASCs(20000, suffix='-HighAltOC', altCarOCValue=-1378/1000, altSUVOCValue=-1438/1000)
