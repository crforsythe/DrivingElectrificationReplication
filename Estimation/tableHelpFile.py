from tqdm import tqdm
from glob import glob
import numpy as np
import pandas as p
from scipy.stats import t as tDist
from scipy.stats import norm as normDist
from copy import deepcopy
from os.path import split
from collections import OrderedDict
from helpFile import testEquality, getModelFiles, getCleanModelInfo
def checkFileRelevance(fileName, disc, antiDisc):
    relevant = True

    for tempDisc in disc:
        if(tempDisc not in fileName):
            relevant = False
    for tempAntiDisc in antiDisc:
        if(tempAntiDisc in fileName):
            relevant = False

    return relevant

def cleanNames(name, space = 'wtp', model='mnl', linearRange=True):
    baseNames = ['price', 'oc', 'acc', 'hev', 'phev10', 'phev20', 'phev40', 'bev75', 'bev100', 'bev150', 'bev250', 'bev300', 'bev400', 'american', 'japanese', 'chinese', 'skorean', 'bevFC', 'phevFC', 'bevRange', 'asinhBEVRange', 'bev', 'bev*AP2eGridDamagesMean', 'bev*co2Mean', 'bevAge', 'asinhBEVRangeAge', 'bevIncome', 'asinhBEVRangeIncome', 'bevWoman', 'asinhBEVRangeWoman', 'pc', 'tc', 'nobevFC', 'BEV_NoFastChargeSD']
    if(space=='wtp'):
        prettyNames = ['Scaling Factor', 'Operating Cost', 'Acceleration', 'HEV', 'PHEV 10', 'PHEV 20', 'PHEV 40', 'BEV 75', 'BEV 100', 'BEV 150',
                       'BEV 250', 'BEV 300', 'BEV 400', 'American', 'Japanese', 'Chinese', 'South Korean',
                       'BEV Fast Charging', 'PHEV Fast Charging', 'BEV Range', 'Arcsinh(BEV Range)', 'BEV', 'BEV*Regional Emissions Damages', 'BEV*Regional CO2 Emissions', 'BEV*(Age-50)', 'Arcsinh(BEV Range)*(Age-50)', 'BEV*(Income-$100k)', 'Arcsinh(BEV Range)*(Income-$100k)', 'BEV*Woman', 'Arcsinh(BEV Range)*Woman', 'Payload Capacity', 'Towing Capacity', 'No BEV Fast Charging', 'No BEV Fast Charging']
    else:
        prettyNames = ['Price', 'Operating Cost', 'Acceleration', 'HEV', 'PHEV 10', 'PHEV 20', 'PHEV 40', 'BEV 75', 'BEV 100', 'BEV 150',
                       'BEV 250', 'BEV 300', 'BEV 400', 'American', 'Japanese', 'Chinese', 'South Korean',
                       'BEV Fast Charging', 'PHEV Fast Charging', 'BEV Range', 'Arcsinh(BEV Range)', 'BEV', 'BEV*Regional Emissions Damages', 'BEV*Regional CO2 Emissions', 'BEV*(Age-50)', 'Arcsinh(BEV Range)*(Age-50)', 'BEV*(Income-$100k)', 'Arcsinh(BEV Range)*(Income-$100k)', 'BEV*Woman', 'Arcsinh(BEV Range)*Woman', 'Payload Capacity', 'Towing Capacity', 'No BEV Fast Charging', 'No BEV Fast Charging']

    mapping = dict(zip(baseNames, prettyNames))

    # print('Name={}'.format(name))

    if(model=='mnl'):
        # if(name in mapping.keys()):
        #     return mapping[name]
        # else:
        #     return name
        for uglyName, prettyName in mapping.items():
            if(uglyName in name):
                if(name.index(uglyName)==0 and (('*' in name)==('*'in prettyName))):
                    name = prettyName

        return name
    else:

        for uglyName, prettyName in mapping.items():
            if(uglyName in name):
                if(name.index(uglyName)==0 and ('*' in name=='*'in prettyName)):
                    name = prettyName
        return name

def getClassifierFromFile(file):
    direc, r = split(file)

    r = r[:r.index('.')]

    return r

def buildTableDF(modelData, modelName, robust=False, mturk=True, python=True, useAbsSD = True):
    params = modelData['params']
    coefs = modelData['coef']
    # modelName = modelData['vehType']
    if('vehData' in modelData.keys()):
        if(python):
            numInds = len(set(modelData['vehData'].loc[:, 'ID']))
            numObs = len(set(modelData['vehData'].loc[:, 'QuestionID']))
        else:
            numInds = modelData['numInds']
            numObs = modelData['numObs']
        modelType = modelData['modelType']
        modelSpace = modelData['space']
        mnl = modelData['modelType']=='mnl'
        ll = modelData['ll']
    else:
        mnl = True
        for param in modelData['params']:
            if('SD' in param):
                mnl=False
    if(robust):
        se = modelData['robustSE']
        t = modelData['robustT']
    else:
        se = modelData['se']
        t = modelData['t']

    if('bestModel' not in modelData.keys()):
        python = False

    if(python and 'pValues' in modelData.keys()):
        pVals = modelData['pValues']
    else:
        pVals = []
        for tempT in t:
            pVals.append(2 * normDist.cdf(-abs(tempT)))
            print('{}:{}'.format(tempT, 2 * normDist.cdf(-abs(tempT))))

    sig = []
    for tempP in pVals:
        if(tempP<.001):
            tempSig = '***'
        elif(tempP<.010):
            tempSig = '**'
        elif(tempP<.050):
            tempSig = '*'
        else:
            tempSig = ''
        sig.append(tempSig)

    prettyCoefs = np.round(coefs, 2)
    prettyNames = []
    strings = []
    distParams = []
    for i in range(len(prettyCoefs)):

        if('SD' in params[i]):
            if(useAbsSD):
                coefs[i] = abs(coefs[i])
            else:
                coefs[i] = coefs[i]

        if(cleanNames(params[i]) in ['Scaling Factor'] or '*' in params[i]):
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
        print(params[i])
        print('Pretty name -{}-'.format(tempPrettyName))
        if('BEV Range' == tempPrettyName):
            # print('scaling the bev range')
            scalingFactor = 100
            tempPrettyName+= " ({}s of Miles)".format(scalingFactor)
            # print('Original BEV Range entry: {}'.format(tempCoefEntry))
            try:
                tempCoefEntry = str(np.round(coefs[i][0] * scalingFactor, roundingDigits)) + sig[i] + ' ({})'.format(
                    np.round(np.sqrt((se[i] ** 2) * (scalingFactor ** 2)), roundingDigits))
            except IndexError:
                tempCoefEntry = str(np.round(coefs[i] * scalingFactor, roundingDigits)) + sig[i] + ' ({})'.format(
                    np.round(np.sqrt((se[i] ** 2) * (scalingFactor ** 2)), roundingDigits))
            # print('BEV Range entry: {}'.format(tempCoefEntry))

        strings.append(tempCoefEntry)
        prettyNames.append(tempPrettyName)
        if(mnl):
            distParams.append('$\mu$')
        else:
            if('SD' in params[i]):
                distParams.append('$\sigma$')
            else:
                distParams.append('$\mu$')


    if('vehData' in modelData.keys()):
        prettyNames.append('\hline Log-likelihood')
        prettyNames.append('Number of Individuals')
        prettyNames.append('Number of Observations')

        if(abs(ll)>100):
            roundDigits = 1
        else:
            roundDigits = 3
        strings.append(str(np.round(ll,roundDigits)))
        strings.append(str(int(numInds)))
        strings.append(str(int(numObs)))

        distParams.extend(['']*3)

    strings = np.array(strings).reshape((-1,1))
    distParams = np.array(distParams).reshape((-1,1))
    prettyNames = np.array(prettyNames).reshape((-1,1))

    df = p.DataFrame(np.hstack((prettyNames, distParams, strings)), columns=['Attribute', 'Parameter',modelName])
    return df

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

def buildTableFromFiles(files, simpleName=False):
    datas = {}

    unweighted = True
    car = False
    suv = False
    truck = False
    pooled = False
    mturk = False
    dynata = False
    wtp = False

    john = False
    pooled = False

    order = ['Acceleration', 'Operating Cost', 'Towing Capacity', 'Payload Capacity', 'HEV', 'PHEV 10', 'PHEV 20', 'PHEV 40', 'PHEV Fast Charging', 'BEV 75', 'BEV 100', 'BEV 150', 'BEV 300', 'BEV 400', 'BEV', 'Arcsinh(BEV Range)', 'BEV Range', 'BEV Range (100s of Miles)', 'BEV*(Age-50)', 'Arcsinh(BEV Range)*(Age-50)', 'BEV*(Income-$100k)', 'Arcsinh(BEV Range)*(Income-$100k)', 'BEV*Woman', 'Arcsinh(BEV Range)*Woman', 'BEV Fast Charging', 'No BEV Fast Charging', 'American', 'Chinese', 'Japanese', 'South Korean', '\hline Log-likelihood', 'Number of Individuals', 'Number of Observations']

    for modelFile in files:
        if('-weighted' in modelFile):
            unweighted=False
        if('-car' in modelFile):
            car = True
        if('-suv' in modelFile):
            suv = True
        if('-wtp' in modelFile):
            wtp = True
        if('john' in modelFile):
            john=True
        if('pooled' in modelFile):
            pooled = True
        elif('mturk' in modelFile):
            mturk = True
        elif('dynata' in modelFile):
            dynata = True
        classifier = getClassifierFromFile(modelFile)
        datas[classifier] = getCleanModelInfo(modelFile)
    dfs = OrderedDict()
    for name, data in datas.items():
        tempDF = buildTableDF(data, cleanModelName(name, john and pooled, simpleName=simpleName))
        dfs[name] = tempDF

    r = dfs[list(dfs.keys())[0]]

    for key in list(dfs.keys())[1:]:
        r = r.merge(dfs[key], on=['Attribute', 'Parameter'], how='outer')

    r.loc[:, 'Order'] = 0

    for orderVar in order:
        r.loc[r.loc[:, 'Attribute']==orderVar, 'Order'] = order.index(orderVar)


    r = r.sort_values(['Order', 'Parameter'])
    r = r.drop('Order', axis=1)
    r = r.replace('nan', '')
    r = r.fillna('')
    r.loc[r.loc[:, 'Parameter']=='$\sigma$', 'Attribute'] = ''



    if(len(set(r.loc[:, 'Parameter']))==2):
        r = r.drop('Parameter', axis=1)


    return r

def shortenTable(tableDF, simple = False):
    rows = len(tableDF)
    varsToDrop = ['HEV', 'American', 'Chinese', 'South Korean', 'Japanese', 'Scaling']
    indsToDrop = []




    for i in range(rows):

        tempVar = tableDF.iloc[i, 0]
        tempMixedCoef = True
        if(tempVar==''):
            tempVar = tableDF.iloc[i-1, 0]

        for tempToDrop in varsToDrop:

            if(tempToDrop in tempVar):
                # print('{}-{}'.format(tempVar, tempToDrop))
                indsToDrop.append(tableDF.index[i])
    # print(indsToDrop)
    tableDF = tableDF.drop(indsToDrop)
    if(simple):
        tableDF = tableDF.append(dict(zip(list(tableDF.columns), ['\hline \makecell[c]{HEV \\& PHEV \\\\ Indicators}', 'Y', 'Y'])), ignore_index=True)
        tableDF = tableDF.append(dict(zip(list(tableDF.columns), ['Brand Indicators', 'Y', 'Y'])),
                                 ignore_index=True)
        phevFastChargeEntires = ["\makecell[c]{PHEV Fast \\\\ Charging Indicator}"]
        for modelName in list(tableDF.columns)[1:]:
            if('2015' in modelName):
                phevFastChargeEntires.append("Y")
            else:
                phevFastChargeEntires.append("N")
    else:
        tableDF = tableDF.append(
            dict(zip(list(tableDF.columns), ['\hline \makecell[c]{HEV \\& PHEV \\\\ Indicators}', '', 'Y', 'Y'])),
            ignore_index=True)
        tableDF = tableDF.append(dict(zip(list(tableDF.columns), ['Brand Indicators', '', 'Y', 'Y'])),
                                 ignore_index=True)
        phevFastChargeEntires = ["\makecell[c]{PHEV Fast \\\\ Charging Indicator}", ""]
        for modelName in list(tableDF.columns)[2:]:
            if ('2015' in modelName):
                phevFastChargeEntires.append("Y")
            else:
                phevFastChargeEntires.append("N")

    tableDF = tableDF.append(dict(zip(list(tableDF.columns), phevFastChargeEntires)),
                             ignore_index=True)

    firstIndices = []
    finalIndices = []
    for ind in tableDF.index:
        tempAtt = tableDF.iloc[ind, 0]
        if('Number' in tempAtt or 'like' in tempAtt):
            finalIndices.append(ind)
        else:
            firstIndices.append(ind)
    firstIndices.extend(finalIndices)
    tableDF = tableDF.loc[firstIndices, :]
    # print(tableDF.columns)
    # print('lol')
    # print(list(tableDF.columns))

    if('Parameter' not in list(tableDF.columns)):
        tableDF.loc[:, 'Parameter'] = '$\mu$'
        # tableDF.loc[:, 'Attribute'] = tableDF[:, 'Attribute']

    sigmaIndices = tableDF.loc[:, 'Parameter'] == '$\sigma$'
    for column in tableDF.columns:
        # print(column)
        tableDF.loc[sigmaIndices, column] = '\cellcolor{lightgray}'+tableDF.loc[sigmaIndices, column]

    return tableDF

def buildStems(files):
    datas = []

    for file in files:
        if('-simple' in file):
            addMeanName = True
        else:
            addMeanName = False
        datas.append(getCleanModelInfo(file, addMeanName=addMeanName))

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



def buildTestInfo(files, meanOnly=False):
    lol = 0
    stems = buildStems(files)
    groupsToTest = {}
    groupsToTest['bev'] = []
    for stem in stems:

        if(stem in ['bev', 'bevRange', 'asinhBEVRange']):
            groupsToTest['bev'].append(stem)
        else:
            groupsToTest[stem] = [stem]

    # print(groupsToTest)
    meanTestResults = {}
    jointTestResults = {}
    if('-simple' in files[0] or '-simple' in files[1]):
        meanOnly = True
    else:
        meanOnly = False

    for name, group in groupsToTest.items():
        meanTestResults[name] = testEquality(files, group, meanOnly=True)
        if(not meanOnly):
            jointTestResults[name] = testEquality(files, group, meanOnly=False)

    if(meanOnly):
        r = {'mean': meanTestResults}
    else:
        r = {'mean': meanTestResults, 'joint': jointTestResults}
    # print('r = {}'.format(r))
    return r

def cleanTestValues(valueDict, testType, simple = False):
    # print(valueDict)
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
    # r = '\multirow{' + str(1) + '}{*}{' + r + '}'

    return r

def buildTestDF(files):

    testInfo = buildTestInfo(files)
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


def combineTables(estTable, testTable):
    testNames = {'mean':'\makecell{Equality of \\\\ Means \\\\ Wald Stat}', 'joint':'\makecell{Equality of \\\\ Distribution \\\\ Wald Stat}'}

    for key, name in testNames.items():
        if(name in testTable.columns):
            estTable.loc[:, name] = ''

    for ind, row in testTable.iterrows():
        for i in estTable.index:
            if(row['Attribute']==estTable.loc[i, 'Attribute']):
                for key, name in testNames.items():
                    if(name in testTable.columns):
                        estTable.loc[i, name] = row[name]
                    if('\hline' not in estTable.loc[i, 'Attribute']):
                        estTable.loc[i, 'Attribute'] = '\hline '+estTable.loc[i, 'Attribute']
    # print(list(estTable.columns))
    # if(len(list(set(list(estTable.loc[:, 'Parameter']))))==1):
    #     estTable = estTable.drop('Parameter', axis=1)

    return estTable
