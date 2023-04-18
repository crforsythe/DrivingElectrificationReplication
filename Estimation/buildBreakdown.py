import matplotlib.pyplot as plt
import numpy as np
import pandas as p
import warnings
warnings.simplefilter('always')
from buildNetWTP import linearCombinationFromModel
from buildPlots import getPairs, getFullData, constructAllDiffDict
from getPythonModels import loadHelvestonModels, loadPooledModels
from tqdm import tqdm

def getMeanNTPDicts(vehType='Car', john=True):
    if(john):
        models = loadHelvestonModels(['-linear', '2010',vehType.lower(), 'mixed'])
    else:
        models = loadPooledModels(['-linear', '2018',vehType.lower(), 'mixed'], ['demo'])
    model = models[list(models.keys())[0]]

    dCars = constructAllDiffDict(getFullData()[vehType])

    r = {}

    for group, specTuple in tqdm(dCars.items()):
        if('current' in group or 'future' in group):
            if('current' in group):
                groupTuple = ('current', group.split('current')[1], specTuple[1])
            else:
                groupTuple = ('future', group.split('future')[1], specTuple[1])
        else:
            groupTuple = ('old', group.split('old')[1], specTuple[1])
        r[groupTuple] = linearCombinationFromModel(model, specTuple[0])
    return r

def compileToDF(johnDict, newDict):
    rows = []

    for classifier, meanNetWTPTuple in johnDict.items():
        tempRow = ['john']
        tempRow.extend(classifier)
        tempRow.append(float(meanNetWTPTuple[0]))
        tempRow.append(float(meanNetWTPTuple[1]))
        rows.append(tempRow)

    rows = np.array(rows)
    # return rows
    print(rows)
    df = p.DataFrame(rows, columns=['model', 'time', 'vehModel', 'modelYear', 'mean', 'var'])

    rows = []

    for classifier, meanNetWTPTuple in newDict.items():
        tempRow = ['new']
        tempRow.extend(classifier)
        tempRow.append(float(meanNetWTPTuple[0]))
        tempRow.append(float(meanNetWTPTuple[1]))
        rows.append(tempRow)

    rows = np.array(rows)

    df = p.concat([df, p.DataFrame(rows, columns=['model', 'time', 'vehModel', 'modelYear', 'mean', 'var'])])
    df.loc[:, 'mean'] = df.loc[:, 'mean'].astype(float)
    df.loc[:, 'var'] = df.loc[:, 'var'].astype(float)
    df.loc[:, 'modelYear'] = df.loc[:, 'modelYear'].astype(int)
    df.loc[:, 'se'] = np.sqrt(df.loc[:, 'var'])

    df.loc[:, 'TimeClean'] = 2012
    df.loc[df.loc[:, 'time']=='current', 'TimeClean'] = 2021




    return df

def getDataFrames():

    jCar = getMeanNTPDicts('Car')
    jSUV = getMeanNTPDicts('SUV')

    pCar = getMeanNTPDicts('Car', john=False)
    pSUV = getMeanNTPDicts('SUV', john=False)

    dCar = compileToDF(jCar, pCar)
    dSUV = compileToDF(jSUV, pSUV)

    r = {'car':dCar, 'suv':dSUV}

    return r

def fixyYLabels(ax):
    yTicks = plt.yticks()
    # yTicks[0] = range(-40, 10, 41)
    newYTicks = []
    newYLabs = []

    for yTick in range(-40, 41, 10):

        newYTicks.append(int(yTick))
        if(int(yTick)==0):
            newYLabs.append('${}'.format(int(yTick)))
        else:
            newYLabs.append('${}k'.format(int(yTick)))

    plt.yticks(newYTicks, newYLabs)

def plotComparison(data, saveFile = None):

    numTimes = len(set(data.loc[:, 'time']))

    data.loc[:, 'numTimes'] = numTimes
    data.loc[:, 'x'] = 0
    data.loc[data.loc[:, 'time'] == 'current', 'x'] = numTimes - 2
    data.loc[data.loc[:, 'time'] == 'future', 'x'] = numTimes - 1
    data.loc[data.loc[:, 'model'] == 'new', 'x'] = data.loc[data.loc[:, 'model'] == 'new', 'x'] + numTimes
    data.loc[:, 'color'] = 'xkcd:red'
    data.loc[data.loc[:, 'model'] == 'new', 'color'] = 'xkcd:green'

    if('Mini' in saveFile):
        print(data)


    x = 5
    gr = (1+np.sqrt(5))/2
    y = x/gr

    fig, ax = plt.subplots(1,1,figsize=(x,y))

    numEntries = len(data)

    if(numEntries==6):
        johnX = 1
        newX = 4
        mid = 2.5
    else:
        johnX = 0
        newX = 2
        mid = 1.5

    johnColor = 'xkcd:red'
    height = 20

    ax = plt.bar(data.loc[:, 'x'], data.loc[:, 'mean'], color=data.loc[:, 'color'], edgecolor='xkcd:black')
    ax = plt.errorbar(x=data.loc[:, 'x'], y=data.loc[:, 'mean'], yerr=2*data.loc[:, 'se'], ls='', marker='', color='xkcd:black')
    ax = plt.text(johnX, height, '2015 Study\nPreferences', ha='center', va='center')
    ax = plt.text(newX, height, '2021 Study\nPreferences', ha='center', va='center')



    data = data.sort_values(['model', 'modelYear'])
    plt.xticks(range(len(data)), data.loc[:, 'modelYear'])
    plt.xlabel('Model Year')
    plt.ylabel('BEV Net WTP Relative to CV')
    fixyYLabels(ax)
    ylim = [-40, 40]#plt.ylim()
    xlim = plt.xlim()
    ax = plt.plot([mid, mid], ylim, color='xkcd:black')
    ax = plt.plot(xlim, [0, 0], color='xkcd:black')

    plt.ylim(ylim)
    plt.xlim(xlim)
    if(saveFile==None):
        plt.show()
    else:
        plt.savefig(saveFile, bbox_inches='tight')

def plotComparisons():
    d = getDataFrames()
    for vehType, df in d.items():
        models = set(df.loc[:, 'vehModel'])
        for model in tqdm(models):
            subData = df.loc[df.loc[:, 'vehModel'] == model, :]
            saveFile = 'Plots/BreakDownPlots/{}.png'.format(model)
            plotComparison(subData, saveFile)



plotComparisons()