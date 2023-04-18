import pickle
import cupy
import pandas as p
import numpy as np
from xlogit import MultinomialLogit, MixedLogit
from getPythonModels import loadPooledModels, loadHelvestonModels
from tqdm import tqdm, trange
from collections import OrderedDict
from warnings import filterwarnings
def getModelSpace(model):
    if('_scale_factor' in model.coeff_names):
        return 'wtp'
    else:
        return 'pref'

def getModelType(model):
    for name in model.coeff_names:
        if('sd' in name):
            return 'mxl'

    return 'mnl'

def alterNames(model):
    rawNames = model.coeff_names
    newNames = []
    sdNames = getSDCols(rawNames)
    for rawName in rawNames:
        newNames.append(updateName(rawName, sdNames))

    return newNames

def updateName(rawName, sdNames):
    if('_scale_factor'==rawName):
        return 'price'

    if('sd.' in rawName):
        return rawName.split('.')[1]+'SD'

    if(rawName in sdNames):
        return rawName+'Mean'

    return rawName

def getSDCols(rawNames):
    r = []

    for rawName in rawNames:
        if('sd.' in rawName):
            r.append(rawName.split('.')[1])

    return r

def translateModelToDict(model):
    r = OrderedDict()

    r['ll'] = model.loglikelihood
    r['params'] = alterNames(model)
    r['coef'] = model.coeff_
    r['gradient'] = np.sum(model.grad_n, axis=0)
    r['gradient_n'] = model.grad_n
    if(model.robust):
        r['robustVC'] = model.covariance
        r['robustSE'] = model.stderr
        r['robustT'] = model.zvalues
    else:
        r['vc'] = model.covariance
        r['se'] = model.stderr
        r['t'] = model.zvalues
    r['space'] = getModelSpace(model)
    r['modelType'] = getModelType(model)
    r['pValues'] = model.pvalues
    r['rawModel'] = model
    return r
def prepareModelsToEnterDict(models):
    if(type(models)!=type([])):
        models = [models]
    bestModel = models[len(models)-1]
    r = translateModelToDict(bestModel)
    r['bestModel'] = bestModel
    r['allModels'] = models

    return r
def getCVSamples(model, k=5):
    data = model['vehData']
    idCol = 'ID'
    ids = list(set(data.loc[:, idCol]))
    np.random.shuffle(ids)
    numIndividuals = len(ids)
    delta = numIndividuals//k
    r = []
    for i in range(k):
        start = i*delta
        end = start+delta
        if(i==k-1):
            end = numIndividuals
        outSample = ids[start:end]
        inSample = ids[:start]
        inSample1 = ids[end:]
        inSample.extend(inSample1)

        r.append({'out':outSample, 'in':inSample})

    return r

def estimateInOutSample(model, sample, name, iteration, nDraws = 10):

    data = model['vehData']
    modelType = model['modelType']
    startVals = model['coef']
    scalingFactorCol = 'Price'
    params = model['specVars']
    weightCol = name.split('-')[len(name.split('-'))-1]
    choiceCol = 'ChoiceInd'
    panelIDCol = 'ID'
    choiceIDCol = 'QuestionID'
    altIDCol = 'Concept'
    mPool = cupy.get_default_memory_pool()
    # cupy.cuda.set_allocator(None)
    # cupy.cuda.set_allocator(None)

    inSample = sample['in']
    outSample = sample['out']
    subDataIn = data.loc[data.loc[:, 'ID'].isin(inSample), :]
    subDataOut = data.loc[data.loc[:, 'ID'].isin(outSample), :]
    if(modelType=='mnl'):
        tempModel = MultinomialLogit()
        randSpec = None
    else:
        tempModel = MixedLogit()
        randSpec = dict(zip(params, ['n']*len(params)))
    if(randSpec==None):
        tempModel.fit(X=subDataIn[params], y=subDataIn[choiceCol], varnames=params, ids=subDataIn[choiceIDCol], scale_factor=subDataIn[scalingFactorCol], alts=subDataIn[altIDCol], init_coeff=startVals, weights=subDataIn[weightCol.capitalize()], maxiter=10000, robust=False, num_hess=False)
        # predictionsIn = tempModel.predict(X=subDataIn[params], varnames=params, ids=subDataIn[choiceIDCol],
        #                                   scale_factor=subDataIn[scalingFactorCol], alts=subDataIn[altIDCol],
        #                                   weights=subDataIn[weightCol.capitalize()], return_proba=True)
        # predictionsOut = tempModel.predict(X=subDataOut[params], varnames=params, ids=subDataOut[choiceIDCol],
        #                                    scale_factor=subDataOut[scalingFactorCol], alts=subDataOut[altIDCol],
        #                                    weights=subDataOut[weightCol.capitalize()], return_proba=True)
    else:
        tempModel.fit(X=subDataIn[params], y=subDataIn[choiceCol], varnames=params, ids=subDataIn[choiceIDCol],
                      scale_factor=subDataIn[scalingFactorCol], alts=subDataIn[altIDCol], init_coeff=startVals,
                      weights=subDataIn[weightCol.capitalize()], maxiter=10000, robust=False, num_hess=False, randvars=randSpec, n_draws=nDraws, halton_opts={'shuffle':True}, batch_size=100, panels=subDataIn[panelIDCol])
    #     predictionsIn = tempModel.predict(X=subDataIn[params], varnames=params, ids=subDataIn[choiceIDCol],
    #                                       scale_factor=subDataIn[scalingFactorCol], alts=subDataIn[altIDCol],
    #                                       weights=subDataIn[weightCol.capitalize()], return_proba=True, n_draws=nDraws, halton_opts={'shuffle':True})
    #     predictionsOut = tempModel.predict(X=subDataOut[params], varnames=params, ids=subDataOut[choiceIDCol],
    #                                        scale_factor=subDataOut[scalingFactorCol], alts=subDataOut[altIDCol],
    #                                        weights=subDataOut[weightCol.capitalize()], return_proba=True, n_draws=nDraws, halton_opts={'shuffle':True})
    #
    # subDataIn.loc[:, 'predictionProb'] = predictionsIn[1].reshape((-1,1))
    # subDataOut.loc[:, 'predictionProb'] = predictionsOut[1].reshape((-1, 1))
    #
    # subDataIn.loc[:, 'predictionProb'] = subDataIn.loc[:, 'predictionProb']
    # subDataOut.loc[:, 'predictionProb'] = subDataOut.loc[:, 'predictionProb']
    #
    # subDataIn.loc[:, 'll'] = subDataIn.loc[:, 'ChoiceInd'] * np.log(subDataIn.loc[:, 'predictionProb'])
    # subDataOut.loc[:, 'll'] = subDataOut.loc[:, 'ChoiceInd'] * np.log(subDataOut.loc[:, 'predictionProb'])
    #
    # subDataIn.loc[:, 'weightedLL'] = subDataIn.loc[:, 'll'] * subDataIn.loc[:, weightCol.capitalize()]
    # subDataOut.loc[:, 'weightedLL'] = subDataOut.loc[:, 'll'] * subDataOut.loc[:, weightCol.capitalize()]
    #
    tempEntry = {'model':tempModel, 'outData':subDataOut, 'inData':subDataIn}
    tempModel = None
    with open('Data/CrossValidation/SubData/{}-{}.dat'.format(name, iteration), 'wb') as file:
        pickle.dump(tempEntry, file)
        file.close()
    mPool = cupy.get_default_memory_pool()
    pinMPool = cupy.get_default_pinned_memory_pool()
    mPool.free_all_blocks()
    pinMPool.free_all_blocks()

    # print('lol = {}'.format(mPool.total_bytes()))
    # print('lol2 = {}'.format(device))
    # print('lol3 = {}'.format(deviceCount))
    # print('lol2 = {}'.format(pinMPool.total_bytes()))


def performCV(models, nDraws=10000, i=0):
    modelIter = tqdm(models.items())
    r = OrderedDict()

    for name, model in modelIter:
        modelIter.set_description('Working on {}...'.format(name))
        samples = getCVSamples(model)
        print(samples)
        estimateInOutSample(model, samples[i], name, i, nDraws=nDraws)
        #for sample in tqdm(samples):
            #estimateInOutSample(model, sample, name, i, nDraws=nDraws)

if __name__=='__main__':
    np.random.seed(22241995)
    filterwarnings('ignore')
    pooledModels = loadPooledModels(disc='suv')
    helvestonModels = loadHelvestonModels(disc='suv')
    for i in range(5):
        print('i={}'.format(i))
        performCV(pooledModels, nDraws=20000, i=i)
    for i in range(5):
        print('i={}'.format(i))
        performCV(helvestonModels, nDraws=20000, i=i)