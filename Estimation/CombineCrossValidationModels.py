import pandas as p
import numpy as np
from glob import glob
import pickle
from xlogit import MultinomialLogit, MixedLogit
from collections import OrderedDict
from tqdm import tqdm
def loadPickledObject(fileName):
    with open(fileName, 'rb') as file:
        r = pickle.load(file)
        file.close()

    return r

def loadAllCrossValidationModels(antiDisc = 'truck'):
    pattern = 'Data/CrossValidation/SubData/p*.dat'
    files = glob(pattern)

    r = OrderedDict()

    for file in sorted(files):
        if(antiDisc not in file):
            r[file] = loadPickledObject(file)

    return r

def getWeightCol(fileName):
    fileSplit = fileName.split('-')

    for splitPortion in fileSplit:
        if('weight' in splitPortion):
            return splitPortion

def estimateInOutSample(fileName, modelDict, nDraws = 10000):


    if('mixed' in fileName):
        modelType = 'mxl'
    else:
        modelType = 'mnl'
    scalingFactorCol = 'Price'
    tempModel = modelDict['model']
    if(modelType=='mnl'):
        params = tempModel.coeff_names[:len(tempModel.coeff_names)-1]
    else:
        params = tempModel.coeff_names[:(len(tempModel.coeff_names) - 1)//2]
        # print(params)
        # params = tempModel.coeff_names[:len(tempModel.coeff_names) - 1]

    choiceCol = 'ChoiceInd'
    panelIDCol = 'ID'
    choiceIDCol = 'QuestionID'
    altIDCol = 'Concept'
    weightCol = getWeightCol(fileName)
    subDataIn = modelDict['inData']
    subDataOut = modelDict['outData']

    if(modelType=='mnl'):
        # tempModel.fit(X=subDataIn[params], y=subDataIn[choiceCol], varnames=params, ids=subDataIn[choiceIDCol], scale_factor=subDataIn[scalingFactorCol], alts=subDataIn[altIDCol], init_coeff=tempModel.coeff_, weights=subDataIn[weightCol.capitalize()], maxiter=10000, robust=False, num_hess=False)
        # predictionsIn = tempModel.predict(X=subDataIn[params], varnames=params, ids=subDataIn[choiceIDCol],
        #                                   scale_factor=subDataIn[scalingFactorCol], alts=subDataIn[altIDCol],
        #                                   weights=subDataIn[weightCol.capitalize()], return_proba=True)
        predictionsOut = tempModel.predict(X=subDataOut[params], varnames=params, ids=subDataOut[choiceIDCol],
                                           scale_factor=subDataOut[scalingFactorCol], alts=subDataOut[altIDCol],
                                           weights=subDataOut[weightCol.capitalize()], return_proba=True)
    else:
        # tempModel.fit(X=subDataIn[params], y=subDataIn[choiceCol], varnames=params, ids=subDataIn[choiceIDCol],
        #               scale_factor=subDataIn[scalingFactorCol], alts=subDataIn[altIDCol], init_coeff=startVals,
                      # weights=subDataIn[weightCol.capitalize()], maxiter=10000, robust=False, num_hess=False, randvars=randSpec, n_draws=nDraws, halton_opts={'shuffle':True}, batch_size=250, panels='ID')
        # predictionsIn = tempModel.predict(X=subDataIn[params], varnames=params, ids=subDataIn[choiceIDCol],
        #                                   scale_factor=subDataIn[scalingFactorCol], alts=subDataIn[altIDCol],
        #                                   weights=subDataIn[weightCol.capitalize()], return_proba=True, n_draws=nDraws, halton_opts={'shuffle':True}, batch_size=1000, panels=subDataIn['ID'])
        predictionsOut = tempModel.predict(X=subDataOut[params], varnames=params, ids=subDataOut[choiceIDCol],
                                           scale_factor=subDataOut[scalingFactorCol], alts=subDataOut[altIDCol],
                                           weights=subDataOut[weightCol.capitalize()], return_proba=True, n_draws=nDraws, halton_opts={'shuffle':True}, batch_size=100, panels=subDataOut['ID'])
    #
    # subDataIn.loc[:, 'predictionProb'] = predictionsIn[1].reshape((-1,1))
    subDataOut.loc[:, 'predictionProb'] = predictionsOut[1].reshape((-1, 1))

    # subDataIn.loc[:, 'predictionProb'] = subDataIn.loc[:, 'predictionProb']
    subDataOut.loc[:, 'predictionProb'] = subDataOut.loc[:, 'predictionProb']

    # subDataIn.loc[:, 'll'] = subDataIn.loc[:, 'ChoiceInd'] * np.log(subDataIn.loc[:, 'predictionProb'])
    subDataOut.loc[:, 'll'] = subDataOut.loc[:, 'ChoiceInd'] * np.log(subDataOut.loc[:, 'predictionProb'])

    # subDataIn.loc[:, 'weightedLL'] = subDataIn.loc[:, 'll'] * subDataIn.loc[:, weightCol.capitalize()]
    subDataOut.loc[:, 'weightedLL'] = subDataOut.loc[:, 'll'] * subDataOut.loc[:, weightCol.capitalize()]
    #
    tempEntry = {'model':tempModel, 'outData':subDataOut, 'inData':subDataIn, 'outLL': np.sum(subDataOut.loc[:, 'weightedLL']), 'inLL': tempModel.loglikelihood}


    return tempEntry

def cleanID(fileName):
    r = fileName.split('/')
    r = r[len(r)-1].split('-')
    r[len(r) - 1] = r[len(r)-1].replace('.dat', '')
    keyVals = ['\makecell{Data\\\\Source}', '\makecell{Vehicle\\\\Type}', '\makecell{Range\\\\Parameterization}', '\makecell{Model\\\\Spec.}', '\makecell{Weight\\\\Year}', '\makecell{Sample}']
    r = dict(zip(keyVals, r))
    # r = p.DataFrame.from_dict(r, orient='index')
    return r

def cleanDF(df):
    print(df.columns)
    numericCols = ['\makecell{Sample}', 'outLL', 'inLL']
    df.loc[:, numericCols] = df.loc[:, numericCols].astype(float)

    vehTypeDict = {'car':'Car', 'suv':'SUV'}
    rangeParamDict = {'asinh':'Asinh', 'linear':'Linear'}
    modelSpecDict = {'mixed':'MXL', 'simple':'MNL'}
    weightDict = {'weight2018':2018}

    replaceCols = ['\makecell{Vehicle\\\\Type}', '\makecell{Range\\\\Parameterization}', '\makecell{Model\\\\Spec.}', '\makecell{Weight\\\\Year}']
    replaceDict = dict(zip(replaceCols, [vehTypeDict, rangeParamDict, modelSpecDict, weightDict]))

    for col, tempDict in replaceDict.items():
        df.loc[:, col] = df.loc[:, col].replace(tempDict)

    renameDict = {'outLL':'\makecell{Mean\\\\Out-of-Sample LL}', 'inLL':'\makecell{Mean\\\\In-Sample LL}'}
    df = df.rename(renameDict, axis=1)
    print(df.columns)
    df = df.drop('\makecell{Sample}', axis=1)
    df = df.sort_values(['\makecell{Model\\\\Spec.}', '\makecell{Vehicle\\\\Type}', '\makecell{Range\\\\Parameterization}'])
    return df

def convertPredictionToDF(resDict):
    r = OrderedDict()
    for fileName, resDict in resDict.items():

        cleanIDVal = cleanID(fileName)
        r[fileName] = cleanIDVal
        try:
            for key, val in resDict.items():
                if('LL' in key):
                    r[fileName][key] = val
        except:
            print(resDict)

    r = p.DataFrame.from_dict(r, orient='index')
    r = cleanDF(r)
    return r
np.random.seed(22241995)
t = loadAllCrossValidationModels()
r = OrderedDict()
for fileName, modelDict in tqdm(t.items()):
    r[fileName] = estimateInOutSample(fileName, modelDict, nDraws=10000)

df = convertPredictionToDF(r)
dfGrouped = df.groupby(['\makecell{Vehicle\\\\Type}', '\makecell{Range\\\\Parameterization}', '\makecell{Model\\\\Spec.}', '\makecell{Weight\\\\Year}'], as_index=False).mean()
dfGrouped = dfGrouped.sort_values(['\makecell{Model\\\\Spec.}', '\makecell{Vehicle\\\\Type}', '\makecell{Range\\\\Parameterization}'])

dfGrouped.to_latex('Tables/CV-Table.tex', index=False, float_format='%.0f', escape=False)