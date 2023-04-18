import pickle
import numpy as np
import pandas as p
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
# from buildTables import checkFileRelevance .
from collections import OrderedDict
from scipy.stats import norm, chi2
from scipy.linalg import block_diag
from copy import deepcopy

def cleanModelName(modelName, addStudyName=False, simpleName=False):
    if('-' not in modelName):
        return modelName
    car = False
    suv = False
    truck = False
    mnl = False
    john = False
    pooled = False
    mturk = False
    dynata = False
    linear = False
    asinh = False

    if('-car' in modelName):
        car = True
    elif('-suv' in modelName):
        suv = True
    elif('-truck' in modelName):
        truck = True

    if('mixed' not in modelName):
        mnl = True

    if ('john' in modelName):
        john = True
    elif ('pooled' in modelName):
        pooled = True
    elif('mturk' in modelName):
        mturk = True
    elif('dynata' in modelName):
        dynata = True

    if('linear' in modelName):
        linear = True
    elif('asinh' in modelName):
        asinh = True

    r = ''

    if(mnl):
        r='\makecell{' #+'Simple Logit'
    else:
        r = '\makecell{' #+ 'Mixed Logit'


    # if(pooled):
    #     r+= '\\\\ Pooled Data'
    # elif(mturk):
    #     r+= '\\\\ MTurk Data'
    # elif(dynata):
    #     r+= '\\\\ Dynata Data'

    if(car):
        r+='\\\\ Car-Buyer Sample'
    elif(suv):
        r+='\\\\ SUV-Buyer Sample'
    elif(truck):
        r += '\\\\ Truck-Buyer Sample'

    if(addStudyName):
        if(john):
            r+= '\\\\ 2015 Study'
        else:
            r+= '\\\\ 2021 Study'

    if(linear):
        r+='\\\\ Linear-in-Range'
    elif(asinh):
        r += '\\\\ Arcsinh-in-Range'

    r+='}'

    if(simpleName):
        if (john):
            r = '\makecell{2015 Study}'
        else:
            r = '\makecell{2021 Study}'


    return r

def checkFileRelevance(fileName, disc, antiDisc):
    relevant = True
    if(type(disc)==str):
        disc = [disc]
    if(type(antiDisc)==str):
        antiDisc = [antiDisc]
    for tempDisc in disc:
        if(tempDisc not in fileName):
            relevant = False
    for tempAntiDisc in antiDisc:
        if(tempAntiDisc in fileName):
            relevant = False

    return relevant

def testVarsInData(datas, vars):
    r = True
    for tempVar in vars:
        for tempData in datas:
            if(tempVar not in tempData['params']):
                return False
    return r
def testEquality(models, vars, meanOnly = True):

    if(type(vars)==str):
        vars = [vars]

    varsToTest = []

    for tempVar in vars:
        varsToTest.append(tempVar+'Mean')
        if(not meanOnly):
            varsToTest.append(tempVar+'SD')

    if(len(models)!=2):
        print("Warning - 2 files were not passed when testing for equality of {}".format(vars))

    datas = deepcopy(list(models.values()))


    # for file in modelFiles:
    #     if('-simple' in file):
    #         addMeanNames = True
    #     else:
    #         addMeanNames = False
    #     datas.append(getCleanModelInfo(file, addMeanName=addMeanNames))

    print('Testing Variables: {}'.format(varsToTest))
    if(not testVarsInData(datas, varsToTest)):
        print('Missing variable in datas')
        return None

    newCov = datas[0]['vc']
    newVarList = datas[0]['params']
    newBeta = datas[0]['coef'].reshape((-1,1))
    for data in datas[1:]:
        newCov = block_diag(newCov, data['vc'])
        newVarList.extend(data['params'])
        betaAddition = data['coef'].reshape((-1,1))
        # print('{}-{}'.format(newBeta.shape, betaAddition.shape))
        newBeta = np.vstack((newBeta, betaAddition))

    newIndices = []
    for tempVar in varsToTest:
        tempStart = 0
        for i in range(len(datas)):
            tempIndex = newVarList.index(tempVar, tempStart)
            newIndices.append(tempIndex)
            tempStart = tempIndex+1
    # print('Indices: {}'.format(newIndices))
    newCov = newCov[newIndices, :]
    newCov = newCov[:, newIndices]
    newBeta = newBeta[newIndices, :]
    # print(newBeta)
    # print(newCov)

    #Contruct R matrix
    rMat = np.zeros((len(varsToTest), len(newIndices)))
    for i in range(len(varsToTest)):
        if('SD' in varsToTest[i]):
            rMat[i, i * 2] = 1
            rMat[i, i * 2 + 1] = -1

            if(newBeta[i*2,0]<0 and newBeta[i * 2+1, 0] > 0):
                rMat[i, i * 2] = -1

            if (newBeta[i * 2, 0] > 0 and newBeta[i * 2 + 1, 0] < 0):
                rMat[i, i * 2] = -1

            # if (newBeta[i * 2+1, 0] < 0):
            #     rMat[i, i * 2+1] = -1

        else:
            rMat[i, i * 2] = 1
            rMat[i, i * 2 + 1] = -1
    # print(rMat)

    outerMat = rMat@newBeta
    innerMat = rMat@newCov@np.transpose(rMat)
    # print(outerMat)
    # print(innerMat)

    chi2Stat = np.transpose(outerMat)@np.linalg.inv(innerMat)@outerMat
    chi2Stat = chi2Stat[0,0]

    # print(chi2Stat)
    pVal = 1-chi2.cdf(chi2Stat, len(varsToTest))
    # print(pVal)
    r = {'stat': chi2Stat, 'p':pVal, 'k':len(varsToTest)}
    return r
def loadPickledObject(file):
    with open(file, 'rb') as f:
        r = pickle.load(f)
        f.close()

    return r

def replaceNames(model):
    preNames = ['Acceleration', 'OpCost', 'bevRangeRel', 'BEVFC', 'PHEVFC', 'BEV_FastCharge', 'PHEV_FastCharge', 'Chinese', 'American', 'SKorean', 'Japanese', 'payloadCapacity', 'towingCapacity']
    postNames = ['acc', 'oc', 'bevRange', 'bevFC', 'phevFC', 'bevFC', 'phevFC', 'chinese', 'american', 'skorean', 'japanese', 'pc', 'tc']
    nameDict = dict(zip(preNames, postNames))

    for i in range(len(model['params'])):
        preName = model['params'][i]

        for preNameSub, postNameSub in nameDict.items():
            if(preNameSub in preName):
                preName = preName.replace(preNameSub, postNameSub)
            else:
                preName = str(preName)

        model['params'][i] = preName

    return model
def getRelevantModels(models, disc, antiDisc, addMeanName=False):
    r = {}
    for modelName, model in models.items():
        if(checkFileRelevance(modelName, disc, antiDisc)):
            r[modelName] = replaceNames(model)

        if(addMeanName and modelName in r.keys()):
            for i in range(len(r[modelName]['params'])):
                if(not('Mean' in r[modelName]['params'][i] or 'SD' in r[modelName]['params'][i])):
                    r[modelName]['params'][i] += 'Mean'


    return r

def loadPooledCovidModels(disc=[], antiDisc=[], addMeanName=False):
    file = 'Models/pooledCovidModels.dat'
    r = loadPickledObject(file)
    return getRelevantModels(r, disc, antiDisc, addMeanName)

def loadPooledDynataModels(disc=[], antiDisc=[], addMeanName=False):
    file = 'Models/dynataModels.dat'
    r = loadPickledObject(file)
    return getRelevantModels(r, disc, antiDisc, addMeanName)

def loadPooledMTurkModels(disc=[], antiDisc=[], addMeanName=False):
    file = 'Models/mturkModels.dat'
    r = loadPickledObject(file)
    return getRelevantModels(r, disc, antiDisc, addMeanName)

def loadPooledModels(disc=[], antiDisc=[], addMeanName=False):
    file = 'Models/pooledModels.dat'

    r = loadPickledObject(file)

    return getRelevantModels(r, disc, antiDisc, addMeanName)

def loadPooledUnweightedModels(disc=[], antiDisc=[], addMeanName=False):
    file = 'Models/pooledUnweightedModels.dat'

    r = loadPickledObject(file)

    return getRelevantModels(r, disc, antiDisc, addMeanName)

def loadHelvestonModels(disc=[], antiDisc=[], addMeanName=False):
    file = 'Models/helvestonModels.dat'

    r = loadPickledObject(file)

    return getRelevantModels(r, disc, antiDisc, addMeanName)

def loadHelvestonMTurkModels(disc=[], antiDisc=[], addMeanName=False):
    file = 'Models/helvestonMTurkModels.dat'

    r = loadPickledObject(file)
    return getRelevantModels(r, disc, antiDisc, addMeanName)

# def loadJohnPooledModels(disc=[], antiDisc=[]):

def changeBEVEntry(table):
    if ('BEV Range (100s of Miles)' in list(table.loc[:, 'Attribute']) or '\\hline BEV Range (100s of Miles)' in list(table.loc[:, 'Attribute'])):
        table.loc[table.loc[:, 'Attribute'] == 'BEV', 'Attribute'] = '300-Mile Range BEV'
        table.loc[table.loc[:, 'Attribute'] == '\\hline BEV', 'Attribute'] = '\\hline 300-Mile Range BEV'
    else:
        table.loc[table.loc[:, 'Attribute'] == 'BEV', 'Attribute'] = '0-Mile Range BEV'
        table.loc[table.loc[:, 'Attribute'] == '\\hline BEV', 'Attribute'] = '\\hline 0-Mile Range BEV'

    return table




def cleanTestValues(valueDict, testType, simple = False):
    print(valueDict)
    if(type(valueDict)==type(None)):
        return '--'
    r = str(np.round(valueDict['stat'],2))
    starDict = OrderedDict()
    starDict[.05] = '*'
    starDict[.01] = '**'
    starDict[.001] = '***'
    stars = ''
    for pVal, starCount in starDict.items():
        if(valueDict['p']<pVal):
            stars = starCount

    r = r+stars

    if(testType=='joint'):
        multiple = 1
    elif(testType=='mean'):
        multiple = 2

    if(simple):
        multiple = 1

    r = '\multirow{'+str(int(valueDict['k']*multiple))+'}{*}{'+r+'}'

    return r
def buildStems(models):
    datas = deepcopy(list(models.values()))

    # for file in files:
    #     if('-simple' in file):
    #         addMeanName = True
    #     else:
    #         addMeanName = False
    #     datas.append(getCleanModelInfo(file, addMeanName=addMeanName))

    paramLists = []

    for data in datas:
        paramLists.append(data['params'])

    separateStems = {}

    for i in range(len(paramLists)):
        tempParamList = paramLists[i]
        separateStems[i] = set([])
        for tempParam in tempParamList:
            if('Mean' in tempParam):
                separateStems[i].add(tempParam[:tempParam.index("Mean")])
            if ('SD' in tempParam):
                separateStems[i].add(tempParam[:tempParam.index("SD")])

    r = separateStems[0] & separateStems[1]

    # print(r)
    return r
def buildTestInfo(models, meanOnly=False):
    lol = 0
    stems = buildStems(models)
    groupsToTest = {}
    groupsToTest['bev'] = []
    for stem in stems:

        if(stem in ['bev', 'bevRange', 'asinhBEVRange']):
            # groupsToTest['bev'].append(stem)
            groupsToTest[stem] = [stem]
        else:
            groupsToTest[stem] = [stem]

    # print(groupsToTest)
    meanTestResults = {}
    jointTestResults = {}
    if('-simple' in list(models.keys())[0] or '-simple' in list(models.keys())[1]):
        meanOnly = True
    else:
        meanOnly = False

    for name, group in groupsToTest.items():
        meanTestResults[name] = testEquality(models, group, meanOnly=True)
        if(not meanOnly):
            jointTestResults[name] = testEquality(models, group, meanOnly=False)

    if(meanOnly):
        r = {'mean': meanTestResults}
    else:
        r = {'mean': meanTestResults, 'joint': jointTestResults}
    # print('r = {}'.format(r))
    return r

def cleanNames(name, space = 'wtp', model='mnl', linearRange=True):
    baseNames = ['price', 'oc', 'acc', 'hev', 'phev10', 'phev20', 'phev40', 'bev75', 'bev100', 'bev150', 'bev250', 'bev300', 'bev400', 'american', 'japanese', 'chinese', 'skorean', 'bevFC', 'phevFC', 'bevRange', 'asinhBEVRange', 'bev', 'bevAge', 'asinhBEVRangeAge', 'bevIncome', 'asinhBEVRangeIncome', 'bevWoman', 'asinhBEVRangeWoman', 'pc', 'tc']
    if(space=='wtp'):
        prettyNames = ['Scaling Factor', 'Operating Cost', 'Acceleration', 'HEV', 'PHEV 10', 'PHEV 20', 'PHEV 40', 'BEV 75', 'BEV 100', 'BEV 150',
                       'BEV 250', 'BEV 300', 'BEV 400', 'American', 'Japanese', 'Chinese', 'South Korean',
                       'BEV Fast Charging', 'PHEV Fast Charging', 'BEV Range (100s of Miles)', 'Arcsinh(BEV Range)', 'BEV', 'BEV*(Age-50)', 'Arcsinh(BEV Range)*(Age-50)', 'BEV*(Income-$100k)', 'Arcsinh(BEV Range)*(Income-$100k)', 'BEV*Woman', 'Arcsinh(BEV Range)*Woman', 'Payload Capacity', 'Towing Capacity']
    else:
        prettyNames = ['Price', 'Operating Cost', 'Acceleration', 'HEV', 'PHEV 10', 'PHEV 20', 'PHEV 40', 'BEV 75', 'BEV 100', 'BEV 150',
                       'BEV 250', 'BEV 300', 'BEV 400', 'American', 'Japanese', 'Chinese', 'South Korean',
                       'BEV Fast Charging', 'PHEV Fast Charging', 'BEV Range (100s of Miles)', 'Arcsinh(BEV Range)', 'BEV', 'BEV*(Age-50)', 'Arcsinh(BEV Range)*(Age-50)', 'BEV*(Income-$100k)', 'Arcsinh(BEV Range)*(Income-$100k)', 'BEV*Woman', 'Arcsinh(BEV Range)*Woman', 'Payload Capacity', 'Towing Capacity']

    mapping = dict(zip(baseNames, prettyNames))

    print('Name={}'.format(name))

    if(model=='mnl'):
        if(name in mapping.keys()):
            return mapping[name]
        else:
            return name
    else:
        for uglyName, prettyName in mapping.items():
            if(uglyName in name):
                if(name.index(uglyName)==0):
                    return prettyName
        return name

def buildTestDF(models):

    testInfo = buildTestInfo(models)
    prettyTestInfo = {}

    for testType, testData in testInfo.items():
        prettyTestInfo[testType] = {}
        for name, values in testData.items():

            if(len(testInfo) == 1):
                simple = True
            else:
                simple = False

            prettyTestInfo[testType][name] = cleanTestValues(values, testType, simple=simple)

    r = p.DataFrame.from_dict(prettyTestInfo)

    prettyNames = []

    for name in r.index:
        prettyNames.append(cleanNames(name))
    # print(prettyNames)
    r.loc[:, 'Attribute'] = prettyNames
    r.loc[:, 'Parameter'] = '$\mu$'

    r = r.rename({'mean':'\makecell{Equality of \\\\ Means \\\\ Wald Stat}', 'joint':'\makecell{Equality of \\\\ Distribution \\\\ Wald Stat}'}, axis=1)

    return r

def joinDicts(johnDict, newDict):
    r = OrderedDict()
    for key, model in johnDict.items():
        r[key] = model

    for key, model in newDict.items():
        if(key in r.keys()):
            r[key+'-2'] = model
        else:
            r[key] = model

    for key, model in r.items():
        if('covid' in key.lower() or 'john' in key.lower()):
            r[key]['file'] = 'John'

    return r

def getGradientNorm(modelData):
    return np.linalg.norm(modelData['gradient'])

def getMultipleGradientNorm(models):
    r = {}

    for modelName, modelData in models.items():
        r[modelName] = getGradientNorm(modelData)

    return r

def getMaxSimpleNorm(models):
    gradNorms = getMultipleGradientNorm(models)

    r = 0
    maxModelName = ''
    for name, norm in gradNorms.items():
        if('simple' in name):
            if(norm > r):
                r = norm
                maxModelName = name


    print(r)
    print(maxModelName)
    return r

def getNonConvergedModels(models, normReq=None):

    gradNorms = getMultipleGradientNorm(models)

    if(type(normReq)==type(None)):
        normReq = getMaxSimpleNorm(models)

    r = {}

    for name, gradNorm in gradNorms.items():
        if(gradNorm>normReq):
            r[name] = models[name]

    return r




# t = loadPooledModels()