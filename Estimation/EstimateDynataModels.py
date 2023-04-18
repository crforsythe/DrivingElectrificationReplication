import pandas as p
import numpy as np
from xlogit import MixedLogit, MultinomialLogit
from scipy.stats.qmc import Halton
from tqdm import trange, tqdm
from collections import OrderedDict
import pickle
import warnings
from cupy.cuda import memory
warnings.simplefilter('ignore')


def loadData(type='car', subset='pooled'):
    file = r"Data\ChoiceData\Dynata\DynataWeighted.xlsx"
    return p.read_excel(file, 'cbcLong-{}'.format(type))


def returnNonNoneEntries(preList):
    r = []

    for entry in preList:
        if(type(entry)!=type(None)):
            r.append(entry)

    return r

def estimateModel(data, vars, randSpec=None, choiceCol = 'ChoiceInd', panelIDCol = 'ID', choiceIDCol = 'QuestionID', altIDCol = 'Concept', weightCol=None, scaleFactor = 'Price', numHess=False, robust=False, nDraws=200, finalNDraws=10000, startVals = None, batchSize=1000, maxIter = 10000, tolOpts={'ftol':1e-100}):



    if(np.ndim(startVals)==2):
        simpleFun = lambda x: estimateModel(data, vars, randSpec, choiceCol, panelIDCol, choiceIDCol, altIDCol,
                                            weightCol, scaleFactor, False, False, nDraws, nDraws, x, batchSize, 2000, {})
        robustFun = lambda x: estimateModel(data, vars, randSpec, choiceCol, panelIDCol, choiceIDCol, altIDCol,
                                            weightCol, scaleFactor, True, True, finalNDraws, finalNDraws, x, batchSize, maxIter, tolOpts)
        nStarts = startVals.shape[0]
        models = [simpleFun(None)]

        for i in trange(nStarts):
            tempStartVals = startVals[i, :]
            try:
                tempModel = simpleFun(tempStartVals)
                models.append(tempModel)
            except np.linalg.LinAlgError:
                models.append(None)
        r = returnNonNoneEntries(models)
        tempBestModel = getBestModel(r)
        r.append(robustFun(tempBestModel.coeff_))
        return r
    else:

        if(randSpec==None):
            model = MultinomialLogit()
            if(weightCol==None):
                if(scaleFactor==None):
                    model.fit(X=data[vars], y=data[choiceCol], varnames=vars, ids=data[choiceIDCol], num_hess=numHess, robust=robust, alts=data[altIDCol], init_coeff=startVals, tol_opts=tolOpts, maxiter=maxIter)
                else:
                    model.fit(X=data[vars], y=data[choiceCol], varnames=vars, ids=data[choiceIDCol], num_hess=numHess, scale_factor=data[scaleFactor],
                              robust=robust, alts=data[altIDCol], init_coeff=startVals, tol_opts=tolOpts, maxiter=maxIter)
            else:
                if (scaleFactor == None):
                    model.fit(X=data[vars], y=data[choiceCol], varnames=vars, ids=data[choiceIDCol], num_hess=numHess,
                              robust=robust, weights=data[weightCol], alts=data[altIDCol], init_coeff=startVals, tol_opts=tolOpts, maxiter=maxIter)
                else:

                    model.fit(X=data[vars], y=data[choiceCol], varnames=vars, ids=data[choiceIDCol], num_hess=numHess,
                              scale_factor=data[scaleFactor],
                              robust=robust, weights=data[weightCol], alts=data[altIDCol], init_coeff=startVals, tol_opts=tolOpts, maxiter=maxIter)
        else:
            model = MixedLogit()
            if (weightCol == None):
                if (scaleFactor == None):
                    model.fit(X=data[vars], y=data[choiceCol], varnames=vars, ids=data[choiceIDCol], num_hess=numHess,
                              robust=robust, randvars=randSpec, alts=data[altIDCol], panels=data[panelIDCol], n_draws=nDraws, halton_opts={'shuffle':True}, init_coeff=startVals, batch_size=batchSize, tol_opts=tolOpts, maxiter=maxIter)
                else:
                    model.fit(X=data[vars], y=data[choiceCol], varnames=vars, ids=data[choiceIDCol], num_hess=numHess,
                              scale_factor=data[scaleFactor],
                              robust=robust, randvars=randSpec, alts=data[altIDCol], panels=data[panelIDCol], n_draws=nDraws, halton_opts={'shuffle':True}, init_coeff=startVals, batch_size=batchSize, tol_opts=tolOpts, maxiter=maxIter)
            else:
                if (scaleFactor == None):
                    model.fit(X=data[vars], y=data[choiceCol], varnames=vars, ids=data[choiceIDCol], num_hess=numHess,
                              robust=robust, weights=data[weightCol], randvars=randSpec, alts=data[altIDCol], panels=data[panelIDCol], n_draws=nDraws, halton_opts={'shuffle':True}, init_coeff=startVals, batch_size=batchSize, tol_opts=tolOpts, maxiter=maxIter)
                else:
                    model.fit(X=data[vars], y=data[choiceCol], varnames=vars, ids=data[choiceIDCol], num_hess=numHess,
                              scale_factor=data[scaleFactor],
                              robust=robust, weights=data[weightCol], randvars=randSpec, alts=data[altIDCol], panels=data[panelIDCol], n_draws=nDraws, halton_opts={'shuffle':True}, init_coeff=startVals, batch_size=batchSize, tol_opts=tolOpts, maxiter=maxIter)

        return model

def getNumParameters(vars, randSpec, scaleFactor):
    numVars = len(vars)
    if (scaleFactor != None):
        numVars += 1

    if(randSpec==None):
        return numVars

    for var in vars:
        if (var in randSpec.keys()):
            numVars += 1
    return numVars


def setLB(numPars, scaleFactor, lb):
    if(lb!=None):
        return lb
    else:
        if(scaleFactor==None):
            return [-1]*numPars
        else:
            lb = [0]
            lb.extend([-5]*(numPars-1))
            return lb

def setUB(numPars, scaleFactor, ub):
    if(ub!=None):
        return ub
    else:
        if(scaleFactor==None):
            return [1]*numPars
        else:
            ub = [1]
            ub.extend([5]*(numPars-1))
            return ub


def getBestModel(models):

    if(type(models)!=type([])):
        models = [models]

    bestLL = -np.inf
    bestModel = None

    for model in models:
        if(model.loglikelihood>bestLL):
            bestLL = model.loglikelihood
            bestModel = model

    return bestModel

def getMixedLB(simpleModel, randVarSpec, multiplier=3):


    if(simpleModel['space']=='wtp'):
        preLB = list(simpleModel['coef']-multiplier*simpleModel['se'])

        lb = preLB[:len(preLB)-1]
        lb.extend([-20] * len(randVarSpec))
        lb.append(preLB[len(preLB)-1])
    else:
        lb = list(simpleModel['coef']-multiplier*simpleModel['se'])
        lb.extend([-5] * len(randVarSpec))

    return lb


def getMixedUB(simpleModel, randVarSpec, multiplier=3):
    if (simpleModel['space'] == 'wtp'):
        preUB = list(simpleModel['coef'] + multiplier * simpleModel['se'])

        ub = preUB[:len(preUB) - 1]
        ub.extend([20] * len(randVarSpec))
        ub.append(preUB[len(preUB) - 1])
    else:
        ub = list(simpleModel['coef'] + multiplier * simpleModel['se'])
        ub.extend([5] * len(randVarSpec))

    return ub




def createStartVals(vars, scaleFactor, randSpec=None, numStarts=100, lb=None, ub=None):
    numPars = getNumParameters(vars, randSpec, scaleFactor)

    haltonSequencer = Halton(numPars)

    haltonVals = haltonSequencer.random(numStarts)

    lb = np.array(setLB(numPars, scaleFactor, lb))
    ub = np.array(setUB(numPars, scaleFactor, ub))

    startVals = lb+haltonVals*(ub-lb)

    return startVals

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

def alterNames(model):
    rawNames = model.coeff_names
    newNames = []
    sdNames = getSDCols(rawNames)
    for rawName in rawNames:
        newNames.append(updateName(rawName, sdNames))

    return newNames

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




data = {'car':loadData('car'), 'suv':loadData('suv')}
carSUVVarsPartworth = ['Acceleration', 'OpCost', 'bev100', 'bev150', 'bev300', 'bev400', 'phev20', 'phev40', 'hev', 'chinese', 'american', 'skorean', 'japanese', 'BEVFC']
carSUVVarsLinear = ['Acceleration', 'OpCost', 'bev', 'bevRangeRel', 'phev20', 'phev40', 'hev', 'chinese', 'american', 'skorean', 'japanese', 'BEVFC']
carSUVVarsAsinh = ['Acceleration', 'OpCost', 'bev', 'asinhBEVRange', 'phev20', 'phev40', 'hev', 'chinese', 'american', 'skorean', 'japanese', 'BEVFC']
carSUVSpecs = {'linear':carSUVVarsLinear}

specs = {'car':carSUVSpecs, 'suv':carSUVSpecs}

weights = {'weight2018':'Weight2018'}

nStarts = 100
haltonSequencer = Halton(len(carSUVVarsLinear)+1)
haltonVals = haltonSequencer.random(nStarts)
randSpec = dict(zip(carSUVVarsLinear, ['n']*len(carSUVVarsLinear)))



scaleVar = 'Price'
choiceCol = 'ChoiceInd'
panelIDCol = 'ID'
altIDCol = 'Concept'
choiceIDCol = 'QuestionID'
weightCol = 'Weight2018'
np.random.seed(1211971)
numDraws = 100
finalNumDraws = 20000
#numDraws = 100
batchSize = 100

r = OrderedDict()
for vehType, vehData in data.items():
    spec = specs[vehType]
    print(vehType)
    for specName, specVars in spec.items():
        for modelType in ['simple', 'mixed']:
            for weightName, weightCol in weights.items():

                modelID = 'dynata-{}-{}-{}-{}'.format(vehType, specName, modelType, weightName)
                if(modelType=='mixed'):
                    randSpec = dict(zip(specVars, ['n']*len(specVars)))
                    startVals = createStartVals(specVars, scaleVar, numStarts=nStarts, lb=getMixedLB(r[modelID.replace('mixed', 'simple')], randSpec), ub=getMixedUB(r[modelID.replace('mixed', 'simple')], randSpec), randSpec=randSpec)
                else:
                    randSpec = None
                    startVals = createStartVals(specVars, scaleVar, numStarts=nStarts)
                print(specName)

                tempModels = estimateModel(vehData, specVars, weightCol=weightCol, startVals=startVals, randSpec=randSpec, batchSize=batchSize, nDraws=numDraws, finalNDraws=finalNumDraws)

                tempEntry = prepareModelsToEnterDict(tempModels)
                tempEntry['vehType'] = vehType
                tempEntry['vehData'] = vehData
                tempEntry['specName'] = specName
                tempEntry['specVars'] = specVars
                tempEntry['startVals'] = startVals
                r[modelID] = tempEntry

                with open('Models/dynataModels.dat', 'wb') as file:
                    pickle.dump(r, file)
                    file.close()

                with open('Models/dynataModels.dat', 'rb') as file:
                    rRead = pickle.load(file)
                    file.close()

    print('-'*100)

