# from buildTables import getCleanModelInfo, cleanNames, getClassifierFromFile
import pandas as p
import numpy as np
from tqdm import tqdm, trange
from collections import OrderedDict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from Inflator import Inflator
from scipy.stats import norm
from copy import deepcopy
from scipy.linalg import block_diag
import pickle
from glob import glob
from multiprocessing import Pool, cpu_count
from scipy.stats import norm as normDist
from scipy.stats import chi2


def cleanNames(name, space='wtp', model='mnl'):
    baseNames = ['price', 'oc', 'acc', 'hev', 'phev10', 'phev20', 'phev40', 'bev75', 'bev100', 'bev150', 'bev250',
                 'bev300', 'bev400', 'american', 'japanese', 'chinese', 'skorean', 'bevFC', 'phevFC', 'bevRange',
                 'asinhBEVRange', 'bev', 'bevAge', 'asinhBEVRangeAge', 'bevIncome', 'asinhBEVRangeIncome', 'bevWoman',
                 'asinhBEVRangeWoman', 'pc', 'tc']
    if (space == 'wtp'):
        prettyNames = ['Scaling Factor', 'Operating Cost', 'Acceleration', 'HEV', 'PHEV 10', 'PHEV 20', 'PHEV 40',
                       'BEV 75', 'BEV 100', 'BEV 150',
                       'BEV 250', 'BEV 300', 'BEV 400', 'American', 'Japanese', 'Chinese', 'South Korean',
                       'BEV Fast Charging', 'PHEV Fast Charging', 'BEV Range', 'Arcsinh(BEV Range)', 'BEV',
                       'BEV*(Age-50)', 'Arcsinh(BEV Range)*(Age-50)', 'BEV*(Income-$100k)',
                       'Arcsinh(BEV Range)*(Income-$100k)', 'BEV*Woman', 'Arcsinh(BEV Range)*Woman', 'Payload Capacity',
                       'Towing Capacity']
    else:
        prettyNames = ['Price', 'Operating Cost', 'Acceleration', 'HEV', 'PHEV 10', 'PHEV 20', 'PHEV 40', 'BEV 75',
                       'BEV 100', 'BEV 150',
                       'BEV 250', 'BEV 300', 'BEV 400', 'American', 'Japanese', 'Chinese', 'South Korean',
                       'BEV Fast Charging', 'PHEV Fast Charging', 'BEV Range', 'Arcsinh(BEV Range)', 'BEV',
                       'BEV*(Age-50)', 'Arcsinh(BEV Range)*(Age-50)', 'BEV*(Income-$100k)',
                       'Arcsinh(BEV Range)*(Income-$100k)', 'BEV*Woman', 'Arcsinh(BEV Range)*Woman', 'Payload Capacity',
                       'Towing Capacity']

    mapping = dict(zip(baseNames, prettyNames))

    if (model == 'mnl'):
        if (name in mapping.keys()):
            return mapping[name]
        else:
            return name
    else:

        for uglyName, prettyName in mapping.items():
            if (uglyName in name):
                if (name.index(uglyName) == 0):
                    return prettyName
        return name


def checkFileRelevance(fileName, disc, antiDisc):
    relevant = True

    for tempDisc in disc:
        if (tempDisc not in fileName):
            relevant = False
    for tempAntiDisc in antiDisc:
        if (tempAntiDisc in fileName):
            relevant = False

    return relevant


def separateModelID(file):
    return file[file.index('-') + 1:file.index('.')]


def getSharedParamaters(modelDatas):
    params = [modelDatas[0]['params'], modelDatas[1]['params']]

    shared = []
    notShared = [[], []]
    for tempParam0 in params[0]:
        if (tempParam0 in params[1]):
            shared.append(tempParam0)

    for tempParam0 in params[0]:
        if (tempParam0 not in shared):
            notShared[0].append(tempParam0)

    for tempParam1 in params[1]:
        if (tempParam1 not in shared):
            notShared[1].append(tempParam1)

    return shared


def getSharedIndices(modelDatas):
    sharedParameters = getSharedParamaters(modelDatas)

    r = OrderedDict()
    for param in sharedParameters:
        prevInd = 0
        r[param] = []
        for modelData in modelDatas:
            tempParams = modelData['params']
            r[param].append(tempParams.index(param) + prevInd)
            prevInd += len(tempParams)

    return r


def getParamNum(modelDatas):
    r = []
    for modelData in modelDatas:
        r.append(len(modelData['params']))

    return r


def appendMeanVector(modelDatas):
    r = list(modelDatas[0]['coef'])
    r.extend(modelDatas[1]['coef'])

    return np.array(r).reshape((-1, 1))


def appendCovarianceMatrices(modelDatas):
    if ('bestModel' in modelDatas[0].keys()):
        return block_diag(modelDatas[0]['vc'], modelDatas[1]['vc'])
    else:
        return block_diag(modelDatas[0]['robustVC'], modelDatas[1]['robustVC'])


def getSDIndices(modelDatas):
    r = []

    for modelData in modelDatas:
        for param in modelData['params']:
            if ('SD' in param):
                r.append(1)
            else:
                r.append(0)

    return r


def appendParams(modelDatas):
    r = []
    for modelData in modelDatas:
        for param in modelData['params']:
            r.append(param)

    return r


def get2015Indices(modelDatas, enforceType=False):
    r = []

    for modelData in modelDatas:
        for param in modelData['params']:
            if ('file' in modelData.keys() and not enforceType):

                if ('Helveston' in modelData['file']):
                    # r.append(1)
                    r.append(0)
                else:
                    # r.append(0)
                    r.append(1)
            else:
                if ('truck' == modelData['vehType']):
                    r.append(0)
                else:
                    r.append(1)

    return r


def buildRVec(modelDatas, param, k=1, enforceType=False, switchIndices=False):
    sharedIndicies = getSharedIndices(modelDatas)[param]
    betaVec = list(appendMeanVector(modelDatas).reshape(-1, ))
    if (switchIndices):
        ind2021 = list(np.array(get2015Indices(modelDatas, enforceType=enforceType)))
    else:
        ind2021 = list(1 - np.array(get2015Indices(modelDatas, enforceType=enforceType)))
    sdInd = getSDIndices(modelDatas)

    r = list(np.zeros((len(ind2021),)))

    sameSign = True
    bothNegative = False

    if (np.sign(sharedIndicies[0]) != np.sign(sharedIndicies[1])):
        sameSign = False

    if (sameSign and np.sign(sharedIndicies[0]) < 0):
        bothNegative = True

    for ind in sharedIndicies:
        if (sdInd[ind] == 0):
            if (ind2021[ind] == 1):
                r[ind] = k + 1
            else:
                r[ind] = -k
        else:

            entry21 = k + 1
            entry15 = -k

            if (ind2021[ind] == 1):
                r[ind] = (entry21 ** 2) * betaVec[ind]
            else:
                r[ind] = (entry15 ** 2) * betaVec[ind]

    return np.array(r).reshape((1, len(ind2021)))


def buildRMat(modelDatas, k=1, enforceType=False, switchIndices=False):
    r = []

    for param in list(getSharedIndices(modelDatas).keys()):
        r.append(buildRVec(modelDatas, param, k, enforceType=enforceType, switchIndices=switchIndices))

    r = np.vstack(tuple(r))

    return r


def buildRVecDiff(modelDatas, param, k=1, enforceType=False, switchIndices=False):
    sharedIndicies = getSharedIndices(modelDatas)[param]
    betaVec = list(appendMeanVector(modelDatas).reshape(-1, ))
    if (switchIndices):
        ind2021 = list(np.array(get2015Indices(modelDatas, enforceType=enforceType)))
    else:
        ind2021 = list(1 - np.array(get2015Indices(modelDatas, enforceType=enforceType)))
    sdInd = getSDIndices(modelDatas)

    r = list(np.zeros((len(ind2021),)))

    sameSign = True
    bothNegative = False

    if (np.sign(betaVec[sharedIndicies[0]]) != np.sign(betaVec[sharedIndicies[1]])):
        sameSign = False

    if (sameSign and np.sign(betaVec[sharedIndicies[0]]) < 0):
        bothNegative = True

    isSD = False

    for ind in sharedIndicies:
        if (sdInd[ind] == 0):
            if (ind2021[ind] == 1):
                r[ind] = 1
            else:
                r[ind] = -1
        else:
            isSD = True
            entry21 = k + 1
            entry15 = -k

            if (ind2021[ind] == 1):
                r[ind] = 1
            else:
                if (sameSign):
                    r[ind] = -1
                else:
                    r[ind] = 1

    mult = 1
    if (bothNegative and isSD):
        mult = -1
    if (not sameSign):
        mult = np.sign(betaVec[sharedIndicies[1]])

    return np.array(r).reshape((1, len(ind2021))) * mult


def buildRMatDiff(modelDatas, k=1, enforceType=False, switchIndices=False):
    r = []

    for param in list(getSharedIndices(modelDatas).keys()):
        r.append(buildRVecDiff(modelDatas, param, k, enforceType=enforceType, switchIndices=switchIndices))

    r = np.vstack(tuple(r))

    return r


def getNewBeta(modelDatas, k=1):
    rMat = buildRMat(modelDatas, k)

    params = list(getSharedIndices(modelDatas).keys())

    betaInit = appendMeanVector(modelDatas)

    r = rMat @ betaInit

    # Revert newly-crafted variance parameter to SDs
    for i in range(len(params)):
        param = params[i]
        if ('SD' in param):
            r[i] = np.sqrt(r[i])

    return r


def getDiffBeta(modelDatas, enforceType=False, switchIndices=False):
    rMat = buildRMatDiff(modelDatas, enforceType=enforceType, switchIndices=switchIndices)

    params = list(getSharedIndices(modelDatas).keys())

    betaInit = appendMeanVector(modelDatas)

    r = rMat @ betaInit

    return r


def getNewCovariance(modelDatas, k=1):
    rMat = buildRMat(modelDatas, k)

    params = list(getSharedIndices(modelDatas).keys())

    newBeta = getNewBeta(modelDatas, k)

    covInit = appendCovarianceMatrices(modelDatas)

    r = rMat @ covInit @ np.transpose(rMat)

    # Revert newly-crafted variance parameter to SDs
    newR = []
    for i in range(len(params)):
        param = params[i]
        if ('SD' in param):

            newR.append(1 / (2 * newBeta[i, 0]))
        else:
            newR.append(1)

    newR = np.diag(newR)

    r = newR @ r @ np.transpose(newR)

    return r


def getDiffCovariance(modelDatas, enforceType=False, switchIndices=False):
    rMat = buildRMatDiff(modelDatas, enforceType=enforceType, switchIndices=switchIndices)

    params = list(getSharedIndices(modelDatas).keys())

    covInit = appendCovarianceMatrices(modelDatas)

    r = rMat @ covInit @ np.transpose(rMat)

    return r


def getNewSE(modelDatas, k=1):
    newCov = getNewCovariance(modelDatas, k)

    r = np.diag(newCov)
    r = np.sqrt(r)

    return r


def getDiffSE(modelDatas, enforceType=False, switchIndices=False):
    newCov = getDiffCovariance(modelDatas, enforceType=enforceType, switchIndices=switchIndices)

    r = np.diag(newCov)
    r = np.sqrt(r)

    return r


def forecast(modelDatas, k=1):
    rBeta = getNewBeta(modelDatas, k)
    rCov = getNewCovariance(modelDatas, k)
    rSE = getNewSE(modelDatas, k)

    rParams = list(getSharedIndices(modelDatas).keys())

    r = OrderedDict()
    r['params'] = rParams
    r['coef'] = rBeta
    r['robustVC'] = rCov
    r['robustSE'] = rSE
    r['robustT'] = rBeta.reshape((-1,)) / rSE.reshape((-1,))

    return r


def getPythonDifference(modelDatas, enforceType=False, switchIndices=False):
    rBeta = getDiffBeta(list(modelDatas.values()), enforceType=enforceType, switchIndices=switchIndices)
    rCov = getDiffCovariance(list(modelDatas.values()), enforceType=enforceType, switchIndices=switchIndices)
    rSE = getDiffSE(list(modelDatas.values()), enforceType=enforceType, switchIndices=switchIndices)

    rParams = list(getSharedIndices(list(modelDatas.values())).keys())

    r = OrderedDict()
    r['params'] = rParams
    r['coef'] = rBeta
    r['robustVC'] = rCov
    r['robustSE'] = rSE
    r['robustT'] = rBeta.reshape((-1,)) / rSE.reshape((-1,))

    return r


def loadForecasts():
    with open('Forecasts/forecasts.dat', 'rb') as tempFile:
        r = pickle.load(tempFile)
        tempFile.close()

    return r


def loadDifferences():
    with open('Differences/differences.dat', 'rb') as tempFile:
        r = pickle.load(tempFile)
        tempFile.close()

    return r


def cleanNames(name, space='wtp', model='mnl', linearRange=True):
    baseNames = ['price', 'oc', 'acc', 'hev', 'phev10', 'phev20', 'phev40', 'bev75', 'bev100', 'bev150', 'bev250',
                 'bev300', 'bev400', 'american', 'japanese', 'chinese', 'skorean', 'bevFC', 'phevFC', 'bevRange',
                 'asinhBEVRange', 'bev', 'bevAge', 'asinhBEVRangeAge', 'bevIncome', 'asinhBEVRangeIncome', 'bevWoman',
                 'asinhBEVRangeWoman', 'pc', 'tc', 'nobevFC', 'BEV_NoFastCharge']
    if (space == 'wtp'):
        prettyNames = ['Scaling Factor', 'Operating Cost', 'Acceleration', 'HEV', 'PHEV 10', 'PHEV 20', 'PHEV 40',
                       'BEV 75', 'BEV 100', 'BEV 150',
                       'BEV 250', 'BEV 300', 'BEV 400', 'American', 'Japanese', 'Chinese', 'South Korean',
                       'BEV Fast Charging', 'PHEV Fast Charging', 'BEV Range', 'Arcsinh(BEV Range)', 'BEV',
                       'BEV*(Age-50)', 'Arcsinh(BEV Range)*(Age-50)', 'BEV*(Income-$100k)',
                       'Arcsinh(BEV Range)*(Income-$100k)', 'BEV*Woman', 'Arcsinh(BEV Range)*Woman', 'Payload Capacity',
                       'Towing Capacity', 'No BEV Fast Charging', 'No BEV Fast Charging']
    else:
        prettyNames = ['Price', 'Operating Cost', 'Acceleration', 'HEV', 'PHEV 10', 'PHEV 20', 'PHEV 40', 'BEV 75',
                       'BEV 100', 'BEV 150',
                       'BEV 250', 'BEV 300', 'BEV 400', 'American', 'Japanese', 'Chinese', 'South Korean',
                       'BEV Fast Charging', 'PHEV Fast Charging', 'BEV Range', 'Arcsinh(BEV Range)', 'BEV',
                       'BEV*(Age-50)', 'Arcsinh(BEV Range)*(Age-50)', 'BEV*(Income-$100k)',
                       'Arcsinh(BEV Range)*(Income-$100k)', 'BEV*Woman', 'Arcsinh(BEV Range)*Woman', 'Payload Capacity',
                       'Towing Capacity', 'No BEV Fast Charging', 'No BEV Fast Charging']

    mapping = dict(zip(baseNames, prettyNames))

    if (model == 'mnl'):

        for uglyName, prettyName in mapping.items():
            if (uglyName in name):
                if (name.index(uglyName) == 0):
                    return prettyName
        return name
    else:
        for uglyName, prettyName in mapping.items():
            if (uglyName in name):
                if (name.index(uglyName) == 0):
                    return prettyName
        return name


def cleanModelName(modelName, addStudyName=False, simpleName=False):
    if ('-' not in modelName):
        return modelName
    car = False
    suv = False
    truck = False
    mnl = False
    helveston = False
    pooled = False
    mturk = False
    dynata = False
    linear = False
    asinh = False

    if ('-car' in modelName):
        car = True
    elif ('-suv' in modelName):
        suv = True
    elif ('-truck' in modelName):
        truck = True

    if ('mixed' not in modelName):
        mnl = True

    if ('helveston' in modelName):
        helveston = True
    elif ('pooled' in modelName):
        pooled = True
    elif ('mturk' in modelName):
        mturk = True
    elif ('dynata' in modelName):
        dynata = True

    if ('linear' in modelName):
        linear = True
    elif ('asinh' in modelName):
        asinh = True

    r = ''

    if (mnl):
        r = '\makecell{'  # +'Simple Logit'
    else:
        r = '\makecell{'  # + 'Mixed Logit'

    if (addStudyName):
        if (helveston):
            r += '2015 Study'
        else:
            r += '2021 Study'

    # if(car):
    #     r+='\\\\ Car-Buyer Sample'
    # elif(suv):
    #     r+='\\\\ SUV-Buyer Sample'
    # elif(truck):
    #     r += '\\\\ Truck-Buyer Sample'

    if (linear):
        r += '\\\\ Linear-in-Range'
    elif (asinh):
        r += '\\\\ Arcsinh-in-Range'

    r += '}'

    if (simpleName):
        if (helveston):
            r = '\makecell{2015 Study}'
        else:
            r = '\makecell{2021 Study}'

    return r


def buildTableDF(modelData, modelName, robust=False, mturk=True, python=True, useAbsSD=True):
    params = modelData['params']
    coefs = modelData['coef']
    # modelName = modelData['vehType']
    if ('vehData' in modelData.keys()):
        if (python):
            numInds = len(set(modelData['vehData'].loc[:, 'ID']))
            numObs = len(set(modelData['vehData'].loc[:, 'QuestionID']))
        else:
            numInds = modelData['numInds']
            numObs = modelData['numObs']
        modelType = modelData['modelType']
        modelSpace = modelData['space']
        mnl = modelData['modelType'] == 'mnl'
        ll = modelData['ll']
    else:
        mnl = True
        for param in modelData['params']:
            if ('SD' in param):
                mnl = False
    if (robust):
        se = modelData['robustSE']
        t = modelData['robustT']
    else:
        se = modelData['se']
        t = modelData['t']

    if ('bestModel' not in modelData.keys()):
        python = False

    if (python):
        pVals = modelData['pValues']
    else:
        pVals = []
        for tempT in t:
            pVals.append(2 * normDist.cdf(-abs(tempT)))

    sig = []
    for tempP in pVals:
        if (tempP < .001):
            tempSig = '***'
        elif (tempP < .010):
            tempSig = '**'
        elif (tempP < .050):
            tempSig = '*'
        else:
            tempSig = ''
        sig.append(tempSig)

    prettyCoefs = np.round(coefs, 2)
    prettyNames = []
    strings = []
    distParams = []
    for i in range(len(prettyCoefs)):

        if ('SD' in params[i]):
            if (useAbsSD):
                coefs[i] = abs(coefs[i])
            else:
                coefs[i] = coefs[i]

        if (cleanNames(params[i]) in ['Scaling Factor']):
            roundingDigits = 3
        else:
            roundingDigits = 2

        try:
            tempCoefEntry = str(np.round(coefs[i][0], roundingDigits)) + sig[i] + ' ({})'.format(
                np.round(se[i], roundingDigits))
        except IndexError:
            tempCoefEntry = str(np.round(coefs[i], roundingDigits)) + sig[i] + ' ({})'.format(
                np.round(se[i], roundingDigits))
        tempPrettyName = cleanNames(params[i], 'mnl', 'mnl')

        if ('BEV Range' == tempPrettyName):

            scalingFactor = 100
            tempPrettyName += " ({}s of Miles)".format(scalingFactor)

            try:
                tempCoefEntry = str(np.round(coefs[i][0] * scalingFactor, roundingDigits)) + sig[i] + ' ({})'.format(
                    np.round(np.sqrt((se[i] ** 2) * (scalingFactor ** 2)), roundingDigits))
            except IndexError:
                tempCoefEntry = str(np.round(coefs[i] * scalingFactor, roundingDigits)) + sig[i] + ' ({})'.format(
                    np.round(np.sqrt((se[i] ** 2) * (scalingFactor ** 2)), roundingDigits))

        strings.append(tempCoefEntry)
        prettyNames.append(tempPrettyName)
        if (mnl):
            distParams.append('$\mu$')
        else:
            if ('SD' in params[i]):
                distParams.append('$\sigma$')
            else:
                distParams.append('$\mu$')

    if ('vehData' in modelData.keys()):
        prettyNames.append('\hline Log-likelihood')
        prettyNames.append('Number of Individuals')
        prettyNames.append('Number of Observations')

        if (abs(ll) > 100):
            roundDigits = 1
        else:
            roundDigits = 3
        strings.append(str(np.round(ll, roundDigits)))
        strings.append(str(int(numInds)))
        strings.append(str(int(numObs)))

        distParams.extend([''] * 3)

    strings = np.array(strings).reshape((-1, 1))
    distParams = np.array(distParams).reshape((-1, 1))
    prettyNames = np.array(prettyNames).reshape((-1, 1))

    df = p.DataFrame(np.hstack((prettyNames, distParams, strings)), columns=['Attribute', 'Parameter', modelName])
    return df


def getTableFromModels(models, changeBEV=True):
    if (len(models) == 0):
        return None

    t = []

    for name, model in models.items():
        t.append(buildTableDF(model, name, False))

    r = t[0]

    if (changeBEV):
        r = changeBEVEntry(r)

    for tempDF in t[1:]:
        if (changeBEV):
            tempDF = changeBEVEntry(tempDF)
        r = r.merge(tempDF, on=['Attribute', 'Parameter'], how='outer')

    preNames = list(r.columns)
    newNames = []

    for preName in preNames:
        newNames.append(cleanModelName(preName))

    r = r.rename(dict(zip(preNames, newNames)), axis=1)

    order = ['Acceleration', 'Operating Cost', 'Towing Capacity', 'Payload Capacity', 'HEV', 'PHEV 10', 'PHEV 20',
             'PHEV 40', 'PHEV Fast Charging', 'BEV 75', 'BEV 100', 'BEV 150', 'BEV 300', 'BEV 400', 'BEV',
             '300-Mile Range BEV', 'BEV Range', 'BEV Range (100s of Miles)', '0-Mile Range BEV',
             'Arcsinh(BEV Range)', 'BEV*(Age-50)',
             'Arcsinh(BEV Range)*(Age-50)', 'BEV*(Income-$100k)', 'Arcsinh(BEV Range)*(Income-$100k)', 'BEV*Woman',
             'Arcsinh(BEV Range)*Woman', 'BEV Fast Charging', 'No BEV Fast Charging', 'American', 'Chinese', 'Japanese',
             'South Korean',
             '\hline Log-likelihood', 'Number of Individuals', 'Number of Observations']

    r.loc[:, 'Order'] = -1

    for orderVar in order:
        r.loc[r.loc[:, 'Attribute'] == orderVar, 'Order'] = order.index(orderVar)

    r = r.sort_values(['Order', 'Parameter'])
    r = r.drop('Order', axis=1)
    r = r.replace('nan', '')
    r = r.fillna('')
    r.loc[r.loc[:, 'Parameter'] == '$\sigma$', 'Attribute'] = ''

    # if (changeBEV):
    #     r = changeBEVEntry(r)

    # if()

    return r


def changeBEVEntry(table):
    if ('BEV Range (100s of Miles)' in list(table.loc[:, 'Attribute']) or '\\hline BEV Range (100s of Miles)' in list(
            table.loc[:, 'Attribute'])):
        table.loc[table.loc[:, 'Attribute'] == 'BEV', 'Attribute'] = '300-Mile Range BEV'
        table.loc[table.loc[:, 'Attribute'] == '\\hline BEV', 'Attribute'] = '\\hline 300-Mile Range BEV'
    else:
        table.loc[table.loc[:, 'Attribute'] == 'BEV', 'Attribute'] = '0-Mile Range BEV'
        table.loc[table.loc[:, 'Attribute'] == '\\hline BEV', 'Attribute'] = '\\hline 0-Mile Range BEV'

    return table


def joinDicts(firstDict, secondDict):
    r = OrderedDict()
    for key, model in firstDict.items():
        r[key] = model

    for key, model in secondDict.items():
        if (key in r.keys()):
            r[key + '-2'] = model
        else:
            r[key] = model

    for key, model in r.items():
        if ('covid' in key.lower() or 'helveston' in key.lower()):
            r[key]['file'] = 'helveston'

    return r


def cleanTestValues(valueDict, testType, simple=False):
    if (type(valueDict) == type(None)):
        return '--'
    r = str(np.round(valueDict['stat'], 2))
    starDict = OrderedDict()
    starDict[.05] = '*'
    starDict[.01] = '**'
    starDict[.001] = '***'
    stars = ''
    for pVal, starCount in starDict.items():
        if (valueDict['p'] < pVal):
            stars = starCount

    r = r + stars

    if (testType == 'joint'):
        multiple = 1
    elif (testType == 'mean'):
        multiple = 2
        multiple = 1

    if (simple):
        multiple = 1

    # r = '\multirow{' + str(int(valueDict['k'] * multiple)) + '}{*}{' + r + '}'
    r = '\multirow{' + str(1) + '}{*}{' + r + '}'

    return r


def buildStems(models):
    datas = deepcopy(list(models.values()))

    paramLists = []

    for data in datas:
        paramLists.append(data['params'])

    separateStems = {}

    for i in range(len(paramLists)):
        tempParamList = paramLists[i]
        separateStems[i] = set([])
        for tempParam in tempParamList:
            if ('Mean' in tempParam):
                separateStems[i].add(tempParam[:tempParam.index("Mean")])
            if ('SD' in tempParam):
                separateStems[i].add(tempParam[:tempParam.index("SD")])

    r = separateStems[0] & separateStems[1]

    return r


def combineTables(estTable, testTable):
    testNames = {'mean': '\makecell{Equality of \\\\ Means \\\\ Wald Stat}',
                 'joint': '\makecell{Equality of \\\\ Distribution \\\\ Wald Stat}'}

    for key, name in testNames.items():
        if (name in testTable.columns):
            estTable.loc[:, name] = ''

    for ind, row in testTable.iterrows():
        for i in estTable.index:
            if (row['Attribute'] == estTable.loc[i, 'Attribute']):
                for key, name in testNames.items():
                    if (name in testTable.columns):
                        estTable.loc[i, name] = row[name]
                    if ('\hline' not in estTable.loc[i, 'Attribute']):
                        estTable.loc[i, 'Attribute'] = '\hline ' + estTable.loc[i, 'Attribute']

    return estTable


def varsInDataTest(datas, vars):
    r = True
    for tempVar in vars:
        for tempData in datas:
            if (tempVar not in tempData['params']):
                return False
    return r


def equalityTest(models, vars, meanOnly=True):
    if (type(vars) == str):
        vars = [vars]

    varsToTest = []

    for tempVar in vars:
        varsToTest.append(tempVar + 'Mean')
        if (not meanOnly):
            varsToTest.append(tempVar + 'SD')

    if (len(models) != 2):
        print("Warning - 2 files were not passed when testing for equality of {}".format(vars))

    datas = deepcopy(list(models.values()))

    # for file in modelFiles:
    #     if('-simple' in file):
    #         addMeanNames = True
    #     else:
    #         addMeanNames = False
    #     datas.append(getCleanModelInfo(file, addMeanName=addMeanNames))

    if (not varsInDataTest(datas, varsToTest)):
        print('Missing variable in datas - {}'.format(varsToTest))
        return None

    newCov = datas[0]['vc']
    newVarList = datas[0]['params']
    newBeta = datas[0]['coef'].reshape((-1, 1))
    for data in datas[1:]:
        newCov = block_diag(newCov, data['vc'])
        newVarList.extend(data['params'])
        betaAddition = data['coef'].reshape((-1, 1))

        newBeta = np.vstack((newBeta, betaAddition))

    newIndices = []
    for tempVar in varsToTest:
        tempStart = 0
        for i in range(len(datas)):
            tempIndex = newVarList.index(tempVar, tempStart)
            newIndices.append(tempIndex)
            tempStart = tempIndex + 1

    newCov = newCov[newIndices, :]
    newCov = newCov[:, newIndices]
    newBeta = newBeta[newIndices, :]

    # Contruct R matrix
    rMat = np.zeros((len(varsToTest), len(newIndices)))
    for i in range(len(varsToTest)):
        if ('SD' in varsToTest[i]):
            rMat[i, i * 2] = 1
            rMat[i, i * 2 + 1] = -1

            if (newBeta[i * 2, 0] < 0 and newBeta[i * 2 + 1, 0] > 0):
                rMat[i, i * 2] = -1

            if (newBeta[i * 2, 0] > 0 and newBeta[i * 2 + 1, 0] < 0):
                rMat[i, i * 2] = -1

            # if (newBeta[i * 2+1, 0] < 0):
            #     rMat[i, i * 2+1] = -1

        else:
            rMat[i, i * 2] = 1
            rMat[i, i * 2 + 1] = -1

    outerMat = rMat @ newBeta
    innerMat = rMat @ newCov @ np.transpose(rMat)

    chi2Stat = np.transpose(outerMat) @ np.linalg.inv(innerMat) @ outerMat
    chi2Stat = chi2Stat[0, 0]

    pVal = 1 - chi2.cdf(chi2Stat, len(varsToTest))

    r = {'stat': chi2Stat, 'p': pVal, 'k': len(varsToTest)}
    return r


def buildTestInfo(models, meanOnly=False):
    lol = 0
    stems = buildStems(models)
    groupsToTest = {}
    groupsToTest['bev'] = []
    for stem in stems:

        if (False):  # stem in ['bev', 'bevRange', 'asinhBEVRange'] and
            # groupsToTest['bev'].append(stem)
            groupsToTest[stem] = [stem]
        else:
            groupsToTest[stem] = [stem]

    print('groups: {}'.format(groupsToTest))
    meanTestResults = {}
    jointTestResults = {}
    if ('-simple' in list(models.keys())[0] or '-simple' in list(models.keys())[1]):
        meanOnly = True
    else:
        meanOnly = False

    for name, group in groupsToTest.items():
        meanTestResults[name] = equalityTest(models, group, meanOnly=True)
        if (not meanOnly):
            jointTestResults[name] = equalityTest(models, group, meanOnly=False)

    if (meanOnly):
        r = {'mean': meanTestResults}
    else:
        r = {'mean': meanTestResults, 'joint': jointTestResults}
    print(r)
    return r


def buildTestDF(models):
    testInfo = buildTestInfo(models)
    prettyTestInfo = {}

    for testType, testData in testInfo.items():
        prettyTestInfo[testType] = {}
        for name, values in testData.items():

            if (len(testInfo) == 1):
                simple = True
            else:
                simple = False

            prettyTestInfo[testType][name] = cleanTestValues(values, testType, simple=simple)

    r = p.DataFrame.from_dict(prettyTestInfo)

    prettyNames = []

    for name in r.index:
        prettyNames.append(cleanNames(name))

    r.loc[:, 'Attribute'] = prettyNames
    r.loc[:, 'Parameter'] = '$\mu$'

    r = r.rename({'mean': '\makecell{Equality of \\\\ Means \\\\ Wald Stat}',
                  'joint': '\makecell{Equality of \\\\ Distribution \\\\ Wald Stat}'}, axis=1)

    return r


def buildFullTable(models, diff=False, enforceType=False, switchIndices=False, dropDiff=True):
    if (diff):
        estTable = buildTableDF(getPythonDifference(models, enforceType=enforceType, switchIndices=switchIndices),
                                'Difference', robust=True, useAbsSD=False)
        # print(estTable)
    else:
        estTable = getTableFromModels(models, changeBEV=False)
    testTable = buildTestDF(models)
    print('***Table***')
    print(testTable.columns)
    print(testTable)
    if (len(set(estTable.loc[:, 'Parameter'])) == 1):
        multinomial = True
    else:
        multinomial = False

    # return estTable, testTable

    testNames = {'mean': '\makecell{Equality of \\\\ Means \\\\ Wald Stat}',
                 'joint': '\makecell{Equality of \\\\ Distribution \\\\ Wald Stat}'}

    for key, name in testNames.items():
        if (name in testTable.columns):
            estTable.loc[:, name] = ''

    for ind, row in testTable.iterrows():
        for i in estTable.index:

            if (cleanNames(row['Attribute']) == estTable.loc[i, 'Attribute']):
                for key, name in testNames.items():
                    if (name in testTable.columns):
                        estTable.loc[i, name] = row[name]
                    if ('\hline' not in estTable.loc[i, 'Attribute']):
                        estTable.loc[i, 'Attribute'] = '\hline ' + estTable.loc[i, 'Attribute']

    estTable = changeBEVEntry(estTable)

    order = ['Acceleration', 'Operating Cost', 'Towing Capacity', 'Payload Capacity', 'HEV', 'PHEV 10', 'PHEV 20',
             'PHEV 40', 'PHEV Fast Charging', 'BEV 75', 'BEV 100', 'BEV 150', 'BEV 300', 'BEV 400', 'BEV',
             '300-Mile Range BEV',
             'Arcsinh(BEV Range)', 'BEV Range', 'BEV Range (100s of Miles)', 'BEV*(Age-50)',
             'Arcsinh(BEV Range)*(Age-50)', 'BEV*(Income-$100k)', 'Arcsinh(BEV Range)*(Income-$100k)', 'BEV*Woman',
             'Arcsinh(BEV Range)*Woman', 'BEV Fast Charging', 'No BEV Fast Charging', 'American', 'Chinese', 'Japanese',
             'South Korean', 'Scaling Factor',
             '\hline Log-likelihood', 'Number of Individuals', 'Number of Observations']

    newCol = []
    orderCol = []
    for ind, row in testTable.iterrows():
        cleanName = cleanNames(ind)
        newCol.append(cleanName)
        orderCol.append(order.index(cleanName))

    testTable.loc[:, 'Attribute'] = newCol
    testTable.loc[:, 'Order'] = orderCol
    testTable.loc[:, 'Attribute'] = testTable.loc[:, 'Attribute'].replace(
        {'BEV Range': 'BEV Range (100s of Miles)'})
    testTable = testTable.sort_values('Order')
    testTable = changeBEVEntry(testTable)

    if (multinomial):
        return testTable.loc[:, ['Attribute', '\makecell{Equality of \\\\ Means \\\\ Wald Stat}']]
    else:
        return testTable.loc[:, ['Attribute', '\makecell{Equality of \\\\ Means \\\\ Wald Stat}',
                                 '\makecell{Equality of \\\\ Distribution \\\\ Wald Stat}']]

    estTable.loc[:, 'Order'] = -1

    for orderVar in order:
        estTable.loc[estTable.loc[:, 'Attribute'] == orderVar, 'Order'] = order.index(orderVar)
        estTable.loc[estTable.loc[:, 'Attribute'] == ('\\hline ' + orderVar), 'Order'] = order.index(orderVar)

    estTable = estTable.sort_values(['Order', 'Parameter'])
    estTable = estTable.drop('Order', axis=1)
    estTable = estTable.replace('nan', '')
    estTable = estTable.fillna('')
    estTable.loc[estTable.loc[:, 'Parameter'] == '$\sigma$', 'Attribute'] = ''

    for col in estTable.columns:
        if (col != 'Parameter' and col != 'Difference'):
            estTable.loc[estTable.loc[:, 'Parameter'] == '$\sigma$', col] = ''

    if (len(set(estTable.loc[:, 'Parameter'])) == 1):
        estTable = fixSingleDifferenceTable(estTable)

    if (dropDiff):
        estTable = estTable.drop(['Difference', 'Parameter'], axis=1)

    return estTable


def fixSingleDifferenceTable(table):
    newAttCol = []

    for ind, row in table.iterrows():
        newAttCol.append(row['Attribute'].replace('\hline', ''))

    colToDrop = None
    for col in table.columns:

        if ('quality' in col):
            colToDrop = col

    table.loc[:, 'Attribute'] = newAttCol

    table = table.drop([colToDrop], axis=1)

    return table


def linearCombination(beta, varCov, r):
    beta = beta.reshape((-1, 1))

    newBetas = r @ beta
    newVarCov = r @ varCov @ np.transpose(r)

    return newBetas, newVarCov


def getParamIndices(paramNames, desiredParams, parSuffix='Mean'):
    r = []
    if (type(desiredParams) == str):
        desiredParams = [desiredParams]

    for param in desiredParams:
        if (param in paramNames):
            r.append(paramNames.index(param))
        elif ((param + parSuffix) in paramNames):
            r.append(paramNames.index(param + parSuffix))

    return r


def getRelevantEstimates(modelData, parNames, parSuffix='Mean'):
    modelParNames = modelData['params']
    paramIndices = getParamIndices(modelParNames, parNames, parSuffix)

    rawBetas = modelData['coef']
    if ('bestModel' in modelData.keys()):
        rawVarCov = modelData['vc']
    else:
        rawVarCov = modelData['robustVC']

    relBetas = rawBetas[paramIndices]
    relVarCov = rawVarCov[paramIndices, :]
    relVarCov = relVarCov[:, paramIndices]

    r = relBetas, relVarCov

    return r


def createRMatrix(paramDict):
    rawDeltas = []
    for param, delta in paramDict.items():
        tempDelta = np.array(delta).reshape((-1, 1))
        rawDeltas.append(tempDelta)

    r = np.hstack(tuple(rawDeltas))

    return r


def cleanParamDict(model, paramDict):
    modelParams = model['params']

    newParamDict = OrderedDict()

    for name, delta in paramDict.items():
        if (name in modelParams or (name + 'Mean') in modelParams):
            newParamDict[name] = delta
        elif (name == 'phevRange'):
            newParamDict['phev40'] = (delta - 20) / (20)
            newParamDict['phev20'] = (40 - delta) / (20)
        else:
            if (delta != 0):
                print('{} not found - dropping from calculation'.format(name))

    return newParamDict


def linearCombinationFromModel(model, paramDict):
    print(paramDict)
    print(model['params'])
    paramDict = cleanParamDict(model, paramDict)

    relBeta, relVarCov = getRelevantEstimates(model, paramDict.keys())
    rMatrix = createRMatrix(paramDict)

    r = linearCombination(relBeta, relVarCov, rMatrix)

    return r


def plotBEVCoefficents(models, names=['Mturk', 'Dynata'], title=None, saveFile=None, show=False, startColors=0,
                       logRange=False, newFig=True):
    pointCol = []
    seCol = []
    modelCol = []
    rangeCol = []
    for j in range(len(models)):
        model = models[j]

        modelName = names[j]
        for i in range(len(model['params'])):
            param = model['params'][i]
            coef = model['coef'][i]
            se = model['robustSE'][i]

            if ('bev' in param and 'FC' not in param and 'SD' not in param):
                if ('Mean' in param):
                    allElectricRange = int(param[3:param.index('Mean')])
                else:
                    allElectricRange = int(param[3:])

                pointCol.append(float(coef))
                seCol.append(float(se))
                modelCol.append(modelName)
                rangeCol.append(allElectricRange)

    pointCol = np.array(pointCol).reshape((-1, 1))
    seCol = np.array(seCol).reshape((-1, 1))
    modelCol = np.array(modelCol).reshape((-1, 1))
    rangeCol = np.array(rangeCol).reshape((-1, 1))

    data = np.hstack((modelCol, rangeCol, pointCol, seCol))
    cols = ['Data Source', 'BEV Range', 'Willingness to Pay', 'SE']

    df = p.DataFrame(data, columns=cols)

    for col in cols[1:]:
        df.loc[:, col] = df.loc[:, col].astype(float)

    df.loc[:, 'Log BEV Range'] = np.log(df.loc[:, 'BEV Range'])

    rangeCounter = Counter(df.loc[:, 'BEV Range'])

    x = 6
    gr = (1 + np.sqrt(5)) / 2
    y = x / gr

    if (newFig):
        fig, ax = plt.subplots(1, 1, figsize=(x, y))

    # ax = sns.barplot(x='BEV Range', y='Willingness to Pay', hue='Data Source', data=df, palette='Paired')
    i = 0
    # df = df.loc[df.loc[:, 'Data Source']=='Dynata Pilot',:]
    df.loc[:, 'lb'] = df.loc[:, 'Willingness to Pay'] - 2 * df.loc[:, 'SE']
    df.loc[:, 'ub'] = df.loc[:, 'Willingness to Pay'] + 2 * df.loc[:, 'SE']

    labeld = True
    for source in sorted(list(set(df.loc[:, 'Data Source']))):
        colors = sns.color_palette('Paired', n_colors=16)
        colors = colors[startColors:startColors + 2]
        color = colors[i]
        subData = df.loc[df.loc[:, 'Data Source'] == source, :]
        width = -10

        if (logRange):
            width = -.05

        coefs = np.array(subData.loc[:, 'Willingness to Pay']).reshape((1, -1))

        if (i == 1):
            width *= -1

        j = 0
        for ind, row in subData.iterrows():
            tempErr = row['SE']
            errKW = {'capsize': 3}
            if (j == 3 and i == 1):
                errKW['label'] = '±2 S.E.'

            if (j == 0):
                label = source
            else:
                label = None

            centered = False

            if (rangeCounter[row['BEV Range']] == 1):
                centered = True

            xVar = 'BEV Range'

            if (logRange):
                xVar = 'Log BEV Range'

            if (centered):
                ax = plt.bar(x=row[xVar], height=row['Willingness to Pay'], color=color, edgecolor='xkcd:black',
                             align='center', width=width, yerr=2 * tempErr, label=label, error_kw=errKW)
            else:
                ax = plt.bar(x=row[xVar], height=row['Willingness to Pay'], color=color, edgecolor='xkcd:black',
                             align='edge', width=width, yerr=2 * tempErr, label=label, error_kw=errKW)
            j += 1
        i += 1

    xlim = plt.xlim()
    ax = plt.plot(xlim, [0, 0], color='xkcd:black')
    plt.xlim(xlim)
    plt.ylim([-40, 10])
    plt.ylabel('Willingness to Pay  ($1,000s)')

    if (logRange):
        plt.xlabel('Log BEV Range')
    else:
        plt.xlabel('BEV Range')
    plt.legend(loc='lower right')
    if (title != None):
        plt.title(title)
    if (show):
        plt.show()

    if (saveFile != None):
        plt.savefig('Plots/BEV Range Plots/' + saveFile, bbox_inches='tight', dpi=500)

    return df


def buildWTPOverRange(model, lower=75, upper=450, delta=1, asinh=False, relRange=300):
    rangeList = np.arange(lower, upper + delta, delta)

    bevName = 'bev'
    if (asinh):
        bevRangeName = 'asinhBEVRange'
        relRange = 0
    else:
        bevRangeName = 'bevRange'

    r = OrderedDict()

    for tempRange in rangeList:
        if (asinh):
            tempRangeVal = np.arcsinh(tempRange)
        else:
            tempRangeVal = tempRange
        tempCoef, tempVar = linearCombinationFromModel(model, {bevName: 1, bevRangeName: tempRangeVal - relRange})
        r[tempRange] = {'Coef': float(tempCoef), 'SE': float(np.sqrt(tempVar))}

    r = p.DataFrame.from_dict(r, orient='index')

    r.loc[:, 'Range'] = r.index

    return r


def plotSingleWTPOverRange(model, name=None, lower=75, upper=450, delta=1, asinh=False, newFigure=False, color=None,
                           addLegend=True, relRange=300):
    if ('linear' in model and asinh):
        print('Model file suggest that the model does not use the asinh formulation of range. Altering command.')
        asinh = False

    if ('asinh' in model and not asinh):
        print('Model file suggest that the model uses the asinh formulation of range. Altering command.')
        asinh = True

    data = buildWTPOverRange(model, lower, upper, delta, asinh)

    x = 6
    gr = (1 + np.sqrt(5)) / 2
    y = x / gr

    if (newFigure):
        fig, ax = plt.subplots(1, 1, figsize=(x, y))

    ax = plt.plot(data.loc[:, 'Range'], data.loc[:, 'Coef'], label=name, color=color, zorder=1)
    ax = plt.fill_between(data.loc[:, 'Range'], data.loc[:, 'Coef'] + 2 * data.loc[:, 'SE'],
                          data.loc[:, 'Coef'] - 2 * data.loc[:, 'SE'], alpha=.25, color=color, zorder=1)
    if (name != None and addLegend):
        plt.legend(loc='lower right')

    if (newFigure):
        plt.show()


def plotSingleWTPOverRange(modelName, model, name=None, lower=75, upper=450, delta=1, asinh=False, newFigure=False,
                           color=None, addLegend=True, relRange=300):
    if ('linear' in model and asinh):
        print('Model file suggest that the model does not use the asinh formulation of range. Altering command.')
        asinh = False

    if ('asinh' in model and not asinh):
        print('Model file suggest that the model uses the asinh formulation of range. Altering command.')
        asinh = True

    data = buildWTPOverRange(model, lower, upper, delta, asinh)

    x = 6
    gr = (1 + np.sqrt(5)) / 2
    y = x / gr

    if (newFigure):
        fig, ax = plt.subplots(1, 1, figsize=(x, y))

    ax = plt.plot(data.loc[:, 'Range'], data.loc[:, 'Coef'], label=name, color=color, zorder=1)
    ax = plt.fill_between(data.loc[:, 'Range'], data.loc[:, 'Coef'] + 2 * data.loc[:, 'SE'],
                          data.loc[:, 'Coef'] - 2 * data.loc[:, 'SE'], alpha=.25, color=color, zorder=1)
    if (name != None and addLegend):
        plt.legend(loc='lower right')

    if (newFigure):
        plt.show()


def plotMultipleWTPOverRange(models, names, lowerPass=None, upperPass=None, delta=1, title=None, saveFile=None,
                             startColors=0, direc=None, show=False, ylim=None, relRange=300):
    x = 6
    gr = (1 + np.sqrt(5)) / 2
    y = x / gr

    fig, ax = plt.subplots(1, 1, figsize=(x, y))

    colors = sns.color_palette()
    i = startColors
    for model, name in zip(models, names):
        if ('2015' in name):
            lowerRange = 75
            upperRange = 150
        else:
            lowerRange = 100
            upperRange = 400

        if (lowerPass != None):
            lowerRange = lowerPass

        if (upperPass != None):
            upperRange = upperPass

        tempColor = colors[i]

        addLegend = True
        if (len(models) == 1):
            addLegend = False

        plotSingleWTPOverRange(model, name, color=tempColor, lower=lowerRange, upper=upperRange, addLegend=addLegend)
        i += 1

    plt.xlabel('BEV Range (Miles)')
    plt.ylabel('Willingness to Pay ($1,000s) \n Relative to Conventional Vehicle')

    if (ylim == None):
        plt.ylim([-40, 5])
    else:
        plt.ylim(ylim)

    xlim = plt.xlim()
    plt.plot(xlim, [0, 0], color='xkcd:black', zorder=0)
    plt.xlim(xlim)

    if (title != None):
        plt.title(title)

    if (saveFile != None):
        if (direc == None):
            plt.savefig('Plots/BEV Range Plots Parameterized/{}.png'.format(saveFile), dpi=300, bbox_inches='tight')
        else:
            plt.savefig('{}/{}.png'.format(direc, saveFile), dpi=300, bbox_inches='tight')
    else:
        if (show):
            plt.show()


def plotMultipleWTPOverRange(models, names, lowerPass=None, upperPass=None, delta=1, title=None, saveFile=None,
                             startColors=0, direc=None, show=False, ylim=None, relRange=300, colorInds=None):
    x = 4
    gr = (1 + np.sqrt(5)) / 2
    y = x / gr

    fig, ax = plt.subplots(1, 1, figsize=(x, y))

    colors = sns.color_palette()
    if (type(colorInds) != type(None)):
        i = 0
    else:
        i = startColors
    modelNames = models.keys()
    models = models.values()
    for modelName, model, name in zip(modelNames, models, names):
        if ('2015' in name):
            lowerRange = 75
            upperRange = 150
        elif ('truck' in name.lower()):
            lowerRange = 250
            upperRange = 400
        elif ('pickup' in name.lower()):
            lowerRange = 250
            upperRange = 400
        elif ('suv' in name.lower()):
            lowerRange = 250
            upperRange = 400
        elif ('car' in name.lower()):
            lowerRange = 250
            upperRange = 400
        else:
            lowerRange = 100
            upperRange = 400

        if (lowerPass != None):
            lowerRange = lowerPass

        if (upperPass != None):
            upperRange = upperPass

        if (type(colorInds) != type(None)):
            tempColor = colors[colorInds[i]]
        else:
            tempColor = colors[i]

        addLegend = True
        if (len(models) == 1):
            addLegend = False

        plotSingleWTPOverRange(modelName, model, name, lower=lowerRange, upper=upperRange, color=tempColor,
                               addLegend=addLegend)
        i += 1

    plt.xlabel('BEV Range (Miles)')
    plt.ylabel('Willingness to Pay ($1,000s) \n Relative to Conventional Vehicle')

    if (ylim == None):
        plt.ylim([-40, 5])
    else:
        plt.ylim(ylim)

    xlim = plt.xlim()
    plt.plot(xlim, [0, 0], color='xkcd:black', zorder=0)
    plt.xlim(xlim)

    if (title != None):
        plt.title(title)

    if (saveFile != None):
        if (direc == None):
            plt.savefig('Plots/BEV Range Plots Parameterized/{}.png'.format(saveFile), dpi=300, bbox_inches='tight')
        else:
            plt.savefig('{}/{}.png'.format(direc, saveFile), dpi=300, bbox_inches='tight')
    else:
        if (show):
            plt.show()


def plotCoefficients(models, names, coefficients, title=None, saveFile=None, startColors=0, ylim=None,
                     saveDirec='Plots', addYLabel=''):
    r = OrderedDict()
    numModels = len(models)
    numCoefficients = len(coefficients)
    height = 1
    heightPerModel = height / numModels
    halfHeightPerModel = heightPerModel / 2
    heightPerCoef = 1.5
    halfHeightPerCoef = heightPerCoef / 2

    modelInfos = list(models.values())
    models = list(models.keys())

    i = 0
    for model, modelInfo, name in zip(models, modelInfos, names):

        j = 0

        for coefficient in coefficients:

            loc = (j * heightPerCoef) + halfHeightPerModel + (i * heightPerModel)

            tempDict = {}
            print(modelInfo)
            coef, var = linearCombinationFromModel(modelInfo, {coefficient: 1})
            cleanName = cleanNames(coefficient)

            if ('Operating' in cleanName):
                cleanName += '\n(¢/mile)'
            elif ('Accel' in cleanName):
                cleanName += '\n(Sec. 0-60)'

            tempDict['coef'] = float(coef)
            tempDict['se'] = np.sqrt(float(var))
            tempDict['name'] = cleanName
            tempDict['study'] = name
            tempDict['loc'] = loc
            r[(name, cleanName)] = tempDict
            j += 1
        i += 1

    r = p.DataFrame.from_dict(r, orient='index')

    x = 5
    gr = (1 + np.sqrt(5)) / 2
    y = x / gr

    fig, ax = plt.subplots(1, 1, figsize=(x, y))

    colors = sns.color_palette()
    i = startColors

    for model, name in zip(models, names):
        tempData = r.loc[r.loc[:, 'study'] == name, :]
        errArgs = {'capsize': .2000}
        if (name == names[0]):
            errArgs = {'label': '±2 S.E.'}
        else:
            errArgs = {}
        tempColor = colors[i]

        if (numModels == 1):
            name = None

        plt.bar(tempData.loc[:, 'loc'], tempData.loc[:, 'coef'], width=heightPerModel, edgecolor='xkcd:black',
                yerr=2 * tempData.loc[:, 'se'], label=name, capsize=5, error_kw=errArgs, color=tempColor)
        i += 1

    newYTicks = []
    newYTickLabels = []

    i = 0

    for coefficient in coefficients:
        cleanName = cleanNames(coefficient)
        if ('Operating' in cleanName):
            cleanName += '\n(¢/mile)'
        elif ('Accel' in cleanName):
            cleanName += '\n(Sec. 0-60)'
        elif ('Capacity' in cleanName):
            cleanName = cleanName.replace(' ', '\n')
            cleanName += "\n'000s Lbs"
        newYTick = halfHeightPerCoef + i * heightPerCoef
        newYTicks.append(newYTick)
        newYTickLabels.append(cleanName)
        i += 1

    plt.xticks(newYTicks, newYTickLabels)
    plt.ylabel('Willingness-to-Pay ($1,000s)' + addYLabel)
    xlim = plt.xlim()

    plt.plot(xlim, [0, 0], color='xkcd:black')
    plt.xlim(xlim)
    plt.ylim(ylim)
    if (numModels == 1):
        plt.legend(loc='upper left')
    else:
        plt.legend(loc='lower right')

    if (title != None):
        plt.title(title)
    if (saveFile != None):
        plt.savefig('{}/{}.png'.format(saveDirec, saveFile), bbox_inches='tight', dpi=300)
    else:
        plt.show()
    return r


def plotHeadToHeadWaterfall(model, paramDict, pricePremium=None, title=None, saveFile=None, show=False, ylim=None,
                            addRebate=False, removeRebate=False, textRightAdjusted=False, aboveLine=True,
                            splitAdjustment=False, direc='Plots', x=None):
    if (x == None):
        x = 4.7
    gr = (1 + np.sqrt(5)) / 2
    gr = 1
    y = x / gr

    fig, ax = plt.subplots(1, 1, figsize=(x, y))
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    phevComparison = 'phevRange' in paramDict.keys()

    colors = sns.color_palette('deep', n_colors=10)
    # colors = sns.color_palette('Paired')

    woCreditColor = colors[0]
    woCreditColor = 'xkcd:blue'

    # colors = sns.color_palette('bright', n_colors=10)
    wCreditColor = colors[9]
    wCreditColor = 'xkcd:violet'
    positiveColor = colors[2]
    negativeColor = colors[3]
    netColor = negativeColor

    # woCreditColor = colors[9]
    # wCreditColor = colors[8]
    # positiveColor = colors[2]
    # negativeColor = colors[4]
    # netColor = colors[7]

    netWTP, netWTPVar = linearCombinationFromModel(model, paramDict)
    netWTP, netWTPVar = float(netWTP), float(netWTPVar)
    netWTPSE = float(np.sqrt(netWTPVar)) * 2

    # if('kona' in saveFile.lower()):
    print('{} - {} ({})'.format(saveFile, netWTP, np.sqrt(netWTPVar)))
    print(paramDict)

    prevTempNet = 0
    minTempNet = 0
    i = 0
    width = .75
    newXTicks = []
    newXLabels = []
    yShift = 1
    yShift = 0.5

    relHEV = False

    for paramName, delta in paramDict.items():

        if ('nobevFC' in paramDict.keys()):
            nobevFC = paramDict['nobevFC']
            if (nobevFC == 0):
                bevFCStatement = ' With\nFast Charging'
            else:
                bevFCStatement = ' Without\nFast Charging'

        if (paramName == 'hev' and delta != 0):
            relHEV = True

        if (delta != 0 or ('Range' in paramName and not phevComparison) or 'nobev' in paramName):
            tempParamDict = {paramName: delta}

            tempNet, tempNetVar = linearCombinationFromModel(model, tempParamDict)
            tempNet, tempNetVar = float(tempNet), float(tempNetVar)
            tempCleanName = cleanNames(paramName)
            tempBottom = prevTempNet

            if ('Capacity' in tempCleanName or 'Cost' in tempCleanName):
                tempCleanName = tempCleanName.replace(' ', '\n')

            if (paramName in ['oc', 'acc', 'tc', 'pc']):
                if (paramName == 'oc'):
                    tempCleanName += '\n({} ¢/mi.)'.format(np.round(delta, 1))
                elif (paramName in ['pc', 'tc']):
                    tempCleanName += '\n({}k Lbs.)'.format(np.round(delta, 1))
                else:
                    tempCleanName = 'Accel.'
                    tempCleanName += '\n({} sec.\n0-60 time)'.format(np.round(delta, 1))
            elif (paramName == 'bevFC' or paramName == 'nobevFC'):
                tempCleanName = tempCleanName.replace('Fast', '\nFast')
                tempCleanName = tempCleanName.replace('Charging', '\nCharging')
            elif (paramName == 'asinhBEVRange'):
                tempCleanName = "BEV\n({} Mile\nRange{})".format(int(np.round(np.sinh(delta), 0)), bevFCStatement)
            elif (paramName == 'bevRange'):
                tempCleanName = 'BEV\n({} Mile\nRange{})'.format(int(delta) + 300, bevFCStatement)
            elif (paramName == 'phevRange'):
                tempCleanName = 'PHEV\n({} Mile\nRange)'.format(int(delta))
            if (paramName not in ['bev', 'hev']):
                newXTicks.append(i)
                newXLabels.append(tempCleanName)

            # ax = plt.bar(i, tempNet, width=width, bottom=tempBottom, label=tempCleanName, zorder=0, color=colors[i], edgecolor='xkcd:black')
            if (paramName not in ['bev', 'hev', 'nobevFC']):
                print('{} - Name'.format(paramName))

                if ('Range' in paramName):
                    tempBottom = 0
                    tempPlotNet = prevTempNet + tempNet
                else:
                    tempPlotNet = tempNet

                if (tempPlotNet < 0):
                    color = negativeColor
                else:
                    color = positiveColor

                ax = plt.bar(i, tempPlotNet, width=width, bottom=tempBottom, zorder=0, color=color,
                             edgecolor='xkcd:black')

            prevTempNet += tempNet
            if (paramName not in ['bev', 'hev', 'nobevFC']):
                if (tempPlotNet < 0):
                    plt.text(i, prevTempNet - yShift, tempCleanName, va='top', ha='center')
                    # plt.text(i, prevTempNet - 1 - tempNet, tempCleanName, va='bottom', ha='center')
                else:
                    plt.text(i, prevTempNet + yShift, tempCleanName, va='bottom', ha='center')
                    # plt.text(i, prevTempNet + 1, tempCleanName, va='bottom', ha='center', color='xkcd:green', fontweight=0)
                    # plt.text(i, prevTempNet - 1 - tempNet, tempCleanName, va='top', ha='center')

            i += 1
            if (paramName not in ['bev', 'hev', 'nobevFC']):
                if (prevTempNet < minTempNet):
                    minTempNet = prevTempNet
                plt.plot([i - 1, i], [prevTempNet, prevTempNet], color='xkcd:black')

    # ax = plt.bar(i, netWTP, color='xkcd:red', zorder=4, label='Net WTP', width=width)
    # ax = plt.scatter(i, netWTP, marker='o', color='xkcd:red', s=50, zorder=4, label='Net WTP')
    ax = plt.scatter(i, netWTP, marker='o', color=netColor, s=50, zorder=4, edgecolors='xkcd:black')
    ax = plt.errorbar(i, netWTP, yerr=netWTPSE, ecolor='xkcd:black', capsize=10, zorder=1, label='±2 S.E.', fmt='none')
    ax = plt.text(i, netWTP + netWTPSE + yShift, "Net", ha='center', va='bottom')
    # ax = plt.errorbar(i, netWTP, yerr=netWTPSE, ecolor='xkcd:black', capsize=20, zorder=1, fmt='none')
    newXTicks.append(i)
    newXLabels.append('Net')
    xlim = plt.xlim()
    tempYlim = plt.ylim()

    plt.plot(xlim, [0, 0], color='xkcd:light grey', alpha=1, zorder=-5)
    # plt.fill_between(xlim, [0, 0], [min(ylim), min(ylim)], color = 'xkcd:light grey', alpha=.75, zorder=-5)

    if (textRightAdjusted):
        textXPlacement = max(xlim)
        haArg = 'right'
    else:
        textXPlacement = min(xlim)
        haArg = 'left'
    if (aboveLine):
        vaArg = 'bottom'
        yShift = 0
    else:
        vaArg = 'top'
        yShift = -0.75

    if (splitAdjustment):
        if (textRightAdjusted):
            altTextXPlacement = min(xlim)
            altHaArg = 'left'
        else:
            altTextXPlacement = max(xlim)
            altHaArg = 'right'
        if (aboveLine):
            altVaArg = 'top'
            altYShift = -0.75
        else:
            altVaArg = 'bottom'
            altYShift = 0

    alphaVal = .9
    zOrderVal = -4

    if (phevComparison):
        powertrainType = 'PHEV'
    else:
        powertrainType = 'BEV'

    if (pricePremium != None):

        if (not (addRebate or removeRebate)):
            plt.plot(xlim, [pricePremium, pricePremium], label='{} Price\nPremium'.format(powertrainType),
                     color=woCreditColor, alpha=alphaVal, zorder=zOrderVal)
            plt.text(textXPlacement, pricePremium + yShift, "{} Price\nPremium".format(powertrainType), ha=haArg,
                     va=vaArg, alpha=alphaVal, zorder=zOrderVal, color=woCreditColor)
        else:
            if (addRebate):
                plt.plot(xlim, [pricePremium, pricePremium], label='{} Price\nPremium'.format(powertrainType),
                         color=woCreditColor, alpha=alphaVal, zorder=zOrderVal)
                plt.plot(xlim, [pricePremium - 7.5, pricePremium - 7.5],
                         label='{} Price Premium\nwith $7,500 Credit'.format(powertrainType),
                         color=wCreditColor, alpha=alphaVal, zorder=zOrderVal)
                if (splitAdjustment):
                    print('lol-it works-{}-{}-{}-{}'.format(altTextXPlacement, altYShift, altHaArg, altVaArg))
                    plt.text(altTextXPlacement, pricePremium + altYShift, "{} Price\nPremium".format(powertrainType),
                             ha=altHaArg, va=altVaArg,
                             color=woCreditColor, alpha=alphaVal, zorder=zOrderVal)
                else:
                    plt.text(textXPlacement, pricePremium + yShift, "{} Price\nPremium".format(powertrainType),
                             ha=haArg, va=vaArg,
                             color=woCreditColor, alpha=alphaVal, zorder=zOrderVal)
                plt.text(textXPlacement, pricePremium - 7.5 + yShift,
                         "{} Price Premium\nwith $7,500 Credit".format(powertrainType), ha=haArg, va=vaArg,
                         color=wCreditColor, alpha=alphaVal, zorder=zOrderVal)

            if (removeRebate):
                if (splitAdjustment):
                    plt.text(altTextXPlacement, pricePremium + altYShift, "{} Price\nPremium".format(powertrainType),
                             ha=altHaArg, va=altVaArg,
                             color=woCreditColor, alpha=alphaVal, zorder=zOrderVal)
                else:
                    plt.text(textXPlacement, pricePremium + yShift, "{} Price\nPremium".format(powertrainType),
                             ha=haArg, va=vaArg,
                             color=woCreditColor, alpha=alphaVal, zorder=zOrderVal)
                plt.plot(xlim, [pricePremium, pricePremium],
                         label='{} Price Premium\nwith $7,500 Credit'.format(powertrainType),
                         color=wCreditColor, alpha=alphaVal, zorder=zOrderVal)
                plt.text(textXPlacement, pricePremium + yShift,
                         "{} Price Premium\nwith $7,500 Credit".format(powertrainType), ha=haArg, va=vaArg,
                         color=wCreditColor, alpha=alphaVal, zorder=zOrderVal)
                plt.text(textXPlacement, pricePremium + 7.5 + yShift, "{} Price\nPremium".format(powertrainType),
                         ha=haArg, va=vaArg, color=woCreditColor, alpha=alphaVal, zorder=zOrderVal)

    plt.xlim(xlim)

    plt.xticks([])
    # plt.xticks(newXTicks, newXLabels)

    plt.ylim(ylim)

    if (minTempNet < min(plt.ylim()) and ylim == None):
        plt.ylim(bottom=minTempNet - 1)

    # plt.legend(bbox_to_anchor=(1.01,1), loc='upper left')
    if (relHEV):
        plt.ylabel('{} WTP ($1k) Relative to HEV'.format(powertrainType))
    else:
        plt.ylabel('{} WTP ($1k) Relative to CV'.format(powertrainType))

    if (('currentLeaf' in saveFile or 'futureLeaf' in saveFile) and False):
        plt.ylabel('')
        yticks = plt.yticks()
        print(yticks)
        plt.yticks(yticks[0], [])
    else:
        newYTicks = []
        newYLabs = []
        for ytick in plt.yticks()[0]:
            newYTicks.append(int(ytick))
            newYLabs.append(int(ytick // 1))
        plt.yticks(newYTicks, newYLabs)

    if (title != None):
        plt.title(title)
    if (show):
        plt.show()

    if (saveFile != None):
        plt.savefig('{}/HeadToHeadWaterfall/'.format(direc) + saveFile, bbox_inches='tight', dpi=300)
    else:
        plt.show()


def plotTimeline(timeDict, model, title=None, saveFile=None):
    df = constructTimelineDF(timeDict, model)
    df = df.sort_values('Year')

    x = 2
    gr = (1 + np.sqrt(5)) / 2
    gr = 1
    y = x * gr

    netWTPColor = 'xkcd:blue'
    errorColor = 'xkcd:light blue'
    premiumColor = 'xkcd:red'

    fig, ax = plt.subplots(1, 1, figsize=(x, y))
    # ax = sns.lineplot(x='Year', y='Mean', data=df, color=netWTPColor, ls='--')
    ax = sns.lineplot(x='Year', y='Premium', data=df, color=premiumColor, ls='--')

    ax = plt.errorbar(x='Year', y='Mean', yerr='Err', color=netWTPColor, data=df,
                      capsize=10, ls='--', label=None)

    for i in range(2):
        if (i == 0):
            ax = sns.scatterplot(x='Year', y='Premium', data=df.loc[df.loc[:, 'Year'] < 2023, :], s=50,
                                 color=premiumColor)
            ax = sns.scatterplot(x='Year', y='Mean', color=netWTPColor, data=df.loc[df.loc[:, 'Year'] < 2023, :], s=50)
        else:
            ax = sns.scatterplot(x='Year', y='Premium', color="xkcd:white", data=df.loc[df.loc[:, 'Year'] > 2023, :],
                                 s=50, edgecolor=premiumColor)
            ax = sns.scatterplot(x='Year', y='Mean', color='xkcd:white', data=df.loc[df.loc[:, 'Year'] > 2023, :], s=50,
                                 edgecolor=netWTPColor)

    plt.ylim([-20, 25])
    ylim = plt.ylim()
    xlim = plt.xlim()

    plt.ylim(ylim)
    plt.xlim([2011, max(df.loc[:, 'Year']) + 1])
    ylim = plt.ylim()
    xlim = plt.xlim()

    if (min(df.loc[:, 'Year']) < 2018):
        x, y = max(xlim) - .01 * (max(xlim) - min(xlim)), max(ylim) - .0 * (max(ylim) - min(ylim))
        ha = 'right'
    else:
        x, y = min(xlim) + .08 * (max(xlim) - min(xlim)), max(ylim) - .1 * (max(ylim) - min(ylim))
        ha = 'left'

    plt.text(x, y, 'BEV\nPrice\nPremium', va='top', ha=ha, color=premiumColor)

    if (min(df.loc[:, 'Year']) < 2018):
        x, y = max(xlim) - .01 * (max(xlim) - min(xlim)), max(ylim) - .73 * (max(ylim) - min(ylim))
    else:
        x, y = min(xlim) + .05 * (max(xlim) - min(xlim)), max(ylim) - .6 * (max(ylim) - min(ylim))
    # plt.fill_between([2022, 2030], y1=[max(ylim)]*2, y2=[min(ylim)]*2, color='xkcd:light gray', zorder=0)

    plt.text(x, y, 'BEV\nNet WTP', va='top', ha=ha, color=netWTPColor)

    if (title != None):
        plt.title(title)

    newXTicks = set(df.loc[:, 'Year'])
    newXTicks.add(2013)
    plt.xticks(list(newXTicks))
    # plt.ylabel('$1,000s')
    plt.ylabel('')
    yticks, ylabs = plt.yticks()

    newLabs = []
    newTicks = []

    for ytick in yticks:

        newYTick = int(ytick)
        newTicks.append(newYTick)
        if (ytick == 0):
            newLabs.append('$0')
        elif (ytick > 0):
            newLabs.append('${}k'.format(newYTick))
        else:
            newLabs.append('-${}k'.format(abs(newYTick)))

    plt.ylim([-22, 25])
    xlim = plt.xlim()
    ax = plt.plot(xlim, [0, 0], color='xkcd:gray', zorder=-5)
    plt.xlim(xlim)
    if (
            'leaf' in saveFile or 'kona' in saveFile):  # in saveFile or 'PHEV' in saveFile or 'xe' in saveFile or 'prime' in saveFile or 'a7' in saveFile or 's60' in saveFile or 's90' in saveFile
        plt.yticks(newTicks, newLabs)
        print(saveFile)
    else:
        plt.yticks(newTicks, [])
    if (saveFile != None):
        plt.savefig('Plots/TimelinePlots/{}.png'.format(saveFile), dpi=300, bbox_inches='tight')
    # plt.show()

    return df


def constructTimelineDF(timeDict, model):
    r = {}
    for year, paramDict in timeDict.items():
        tempPrice = paramDict[0]
        tempMean, tempVar = linearCombinationFromModel(model, paramDict[1])
        tempMean, tempVar = float(tempMean), float(tempVar)
        tempNetDict = dict(zip(['Mean', 'Variance', 'Premium'], (tempMean, tempVar, tempPrice)))
        r[year] = tempNetDict

    r = p.DataFrame.from_dict(r)
    r = r.transpose()

    r.loc[:, 'Year'] = r.index
    r.loc[:, 'Std. Err.'] = np.sqrt(r.loc[:, 'Variance'])
    r.loc[:, 'Err'] = 2 * r.loc[:, 'Std. Err.']
    r.loc[:, 'Lower'] = r.loc[:, 'Mean'] - 2 * r.loc[:, 'Std. Err.']
    r.loc[:, 'Upper'] = r.loc[:, 'Mean'] + 2 * r.loc[:, 'Std. Err.']
    r.loc[:, 'Premium'] = r.loc[:, 'Premium'] / 1000
    return r


def buildInnerMatrix(model, clusterCol = 'ID', data=None):
    grad_n = model.grad_n
    grad = list(np.sum(grad_n, axis=0))
    if(type(data)==type(None)):
        return None
    df = p.DataFrame(grad_n)

    subData = data.loc[data.loc[:, 'Concept']==1, :]
    df.loc[:, 'ID'] = list(subData.loc[:, clusterCol])

    for i in range(len(grad)):
        df.loc[:, i] = df.loc[:, i]-(grad[i]/len(subData))

    df = df.groupby('ID').sum()

    r = np.array(df)

    r = np.transpose(r)@r

    return r

def buildOuterMatrix(model):

    return model.hess_inv

def calculateClusterVarCov(model, clusterCol = 'ID', data=None):
    outer = buildOuterMatrix(model)
    inner = buildInnerMatrix(model, clusterCol, data)

    r = outer@inner@outer
    r = np.multiply(r, getCorrection())
    return r

def getCorrection(data):
    n = len(set(data.loc[:, 'ID']))
    r = n/(n-1)
    return r

def getSE(vCov):
    return np.sqrt(np.diag(vCov))

def clusterSEs(model, clusterCol = 'ID', data=None):
    clusterVarCov = calculateClusterVarCov(model, clusterCol, data)
    r = deepcopy(model)
    r.covariance = clusterVarCov
    r.stderr = getSE(clusterVarCov)
    r.zvalues = r.coeff_/r.stderr

    return r