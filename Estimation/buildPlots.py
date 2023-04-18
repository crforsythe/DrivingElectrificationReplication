import matplotlib.pyplot as plt
import pandas as p
import numpy as np
from Inflator import Inflator
from collections import OrderedDict
from buildNetWTP import plotHeadToHeadWaterfall, plotTimeline, constructTimelineDF, linearCombinationFromModel
from helpFile import cleanNames
from getPythonModels import loadPooledModels
import matplotlib.pyplot as plt
from GetPHEVTripDist import getPHEVFuelShare
def getFullData():
    file = 'Data/Spec Data/HeadToHeadSpecsV5.xlsx'
    data = p.read_excel(file, None)
    return data

def getFuelPrice(type='gas', year=2013):
    data = getFullData()
    fuelPriceData = data['FuelCost']

    fuelPriceData.loc[:, 'yearDiff'] = np.abs(fuelPriceData.loc[:, 'year']-year)
    fuelPriceData = fuelPriceData.sort_values('yearDiff')
    fuelPriceData = fuelPriceData.loc[fuelPriceData.loc[:, 'type']==type, :]


    return float(fuelPriceData.iloc[0,0])

def getFuelPriceCorrection(type='gas', year=2013):
    return (getFuelPrice(type, 2022)/getFuelPrice(type, year))

def constructSpecDict(row, relRange=300, truck=False):
    ocAvail = row['oc']==row['oc']
    # print(row)
    r = OrderedDict()
    r['hev'] = int(row['type'] == 'hev')
    r['bev'] = int(row['type']=='bev')
    r['nobevFC'] = row['nobevFC']
    if(r['bev']==1):
        r['bevRange'] = row['range']
    elif(row['type']=='phev'):
        r['phevRange'] = row['range']
        r['bevRange'] = relRange
    else:
        r['bevRange'] = relRange

    if(row['year']<2022):
        infl = Inflator()
        r['price'] = infl.inflateAll(row['price'], row['year'], 2021)
    else:
        r['price'] = row['price']

    if(ocAvail):
        r['oc'] = row['oc']*getFuelPriceCorrection(row['fuelType'], row['year'])
    else:
        if(row['type']=='bev'):
            if(row['fuelEconE']==row['fuelEconE']):
                r['oc'] = 33.705*getFuelPrice(row['fuelType'], 2021)/row['fuelEconE']
            else:
                r['oc'] = row['battSize']/row['range']*getFuelPrice(row['fuelType'], 2021)
        elif(row['type']=='phev'):
            fuelShareElec = getPHEVFuelShare(row['range'])
            gasOC = getFuelPrice('gas', 2021)/row['fuelEcon']
            elecOC = 33.705*getFuelPrice('elec', 2021)/row['fuelEconE']
            r['oc'] = 1/((fuelShareElec/elecOC)+((1-fuelShareElec)/gasOC))
        else:
            try:
                r['oc'] = getFuelPrice(row['fuelType'], 2021)/row['fuelEcon']
            except IndexError:
                print('{} has undefined Op Cost'.format(row['model']))

    if(truck):
        r['tc'] = row['tc']
        r['pc'] = row['pc']
    else:
        r['acc'] = row['acc']


    return r

def getAllKeys(dicts):
    r = []
    for key in dicts[0].keys():
        r.append(key)

    for key in dicts[1].keys():
        if(key not in r):
            r.append(key)

    if('price' in r):
        r.remove('price')

    return r

def constructDiffDict(data, truck=False):
    if(len(data)!=2):
        return None
    else:
        tempR = []
        data = data.sort_values('order')
        for ind, row in data.iterrows():
            tempR.append(constructSpecDict(row, truck=truck))

        allKeys = getAllKeys(tempR)
        r = OrderedDict()
        for key in allKeys:
            if('oc' in key or 'acc' in key or 'price' in key):
                diff = tempR[0][key]-tempR[1][key]
                base = tempR[1][key]
                percDiff = diff/base
            if(key not in tempR[0].keys()):
                r[key] = -tempR[1][key]
            elif(key not in tempR[1].keys()):
                r[key] = tempR[0][key]
            else:
                r[key] = tempR[0][key]-tempR[1][key]

        return r


def getPriceDifferential(data):
    data = data.sort_values('order')
    # print(data)
    priceDiff = float(data.loc[data.loc[:, 'order']==1, 'price'])-float(data.loc[data.loc[:, 'order']==2, 'price'])
    priceBase = float(data.loc[data.loc[:, 'order']==2, 'price'])
    pricePercDiff = priceDiff/priceBase



    if (np.mean(data.loc[:, 'year']) < 2022):
        infl = Inflator()
        priceDiff = infl.inflateAll(priceDiff, int(np.mean(data.loc[:, 'year'])), 2021)
    return priceDiff

def constructAllDiffDict(data, truck=False):
    groups = list(set(data.loc[:, 'groupName']))
    r = OrderedDict()
    for group in groups:
        subData = data.loc[data.loc[:, 'groupName']==group, :]
        try:
            r[group] = (constructDiffDict(subData, truck=truck), int(np.average(subData.loc[:, 'year'])), getPriceDifferential(subData))
        except ValueError:
            print('{} is undefined'.format(group))


    return r

def getFormat(groupName):
    rawData = getFullData()
    data = rawData['format']

    subData = data.loc[data.loc[:, 'groupName']==groupName, :]

    r = {}
    for column in subData.columns:
        for ind, row in subData.iterrows():
            r[column] = row[column]

    return r

def getGroups(type='Car'):
    data = getFullData()[type]
    return list(set(data.loc[:, 'groupName']))

def plotAllHeadToHeads(type='Car'):
    truck = False
    if(type=='Car'):
        models = loadPooledModels(['2018', '-car', '-linear'], ['-demo', '-base'])

    elif (type == 'SUV'):
        models = loadPooledModels(['2018', '-suv', '-linear'], ['-demo', '-base'])
    elif(type=='Truck'):
        models = loadPooledModels(['2018', '-truck', '-linear'], ['-demo', '-base'])
        truck = True
    data = getFullData()[type]

    if(truck):
        x = 5.5
    else:
        x = None

    for modelName, model in models.items():
        diffDicts = constructAllDiffDict(data, truck=truck)
        simData = loadSimResults(modelName)
        for groupName, diffTuples in diffDicts.items():
            diffDict = diffTuples[0]
            year = diffTuples[1]
            priceDiff = diffTuples[2]
            formatDict = getFormat(groupName)
            if(year>2023):
                title = 'Future'
            else:
                title = str(year)

            if(type=='truck' or type=='Truck'):
                title = None
                medianAdopt = np.median(simData.loc[:, groupName]) * 100
                netText = 'Median\nShare:\n{:.0f}%'.format(medianAdopt)
            else:
                netText = None

            plotHeadToHeadWaterfall(model, diffDict, title=title, saveFile=formatDict['saveFile']+modelName+'-'+groupName+'.png', show=formatDict['show'], ylim=(formatDict['ylimMin'], formatDict['ylimMax']),
                                    pricePremium=priceDiff / 1000, addRebate=formatDict['addRebate'], aboveLine=formatDict['aboveLine'],
                                    textRightAdjusted=formatDict['textRightAdjusted'], direc=formatDict['direc'], x=x, netText=netText)


def loadSimResults(modelName):
    print('Loading simulations...')
    file = 'HeadToHeadSims/{}.csv'.format(modelName)
    data = p.read_csv(file)
    print(modelName)
    return data

def getPairs(data):
    groups = tuple(sorted(list(set(data.loc[:, 'groupName']))))

    pairSet = set([])

    for group in groups:
        subData = data.loc[data.loc[:, 'groupName']==group, :]
        pairSet.add(tuple(sorted(list(set(subData.loc[:, 'model'])))))

    return pairSet

def plotAllTimeslines(type = 'Car', bar=False, width=1):
    data = getFullData()[type]

    pairs = getPairs(data)



    if (type == 'Car'):
        models = loadPooledModels(['2018', '-car', '-linear'], ['-demo', '-base'])
    elif(type=='SUV'):

        models = loadPooledModels(['2018', '-suv', '-linear'], ['-demo', '-base'])


    for modelName, tempModel in models.items():
        for pair in pairs:
            subData = data.loc[data.loc[:, 'model'].isin(list(pair)), :]
            diffDicts = constructAllDiffDict(subData)

            tempDict = {}
            saveFile = modelName+'-'+pair[0]+'-'+pair[1]+'-time'

            if(bar):
                for name, diffDict in diffDicts.items():
                    tempDict[diffDict[1]] = (diffDict[2], diffDict[0])
                tempDF = constructTimelineDF(tempDict, tempModel)
                return tempDF
            else:
                for name, diffDict in diffDicts.items():
                    tempDict[diffDict[1]] = (diffDict[2], diffDict[0])
                plotTimeline(tempDict, tempModel, saveFile=saveFile)


def plotTimelineBars(timelineData):
    netWTPColor = 'xkcd:blue'
    errorColor = 'xkcd:light blue'

    x = 2
    gr = (1 + np.sqrt(5)) / 2
    y = x * gr

    fig, ax = plt.subplots(1,1,figsize=(x,y))


    for ind, row in timelineData.iterrows():
        ax = plt.bar(row['Year'], )

if __name__=='__main__':
    # data = getFullData()['SUV']
    # m = loadPooledModels(['2018', 'mixed', 'linear', 'suv'])
    # m = m[list(m.keys())[0]]
    # r = constructAllDiffDict(data)
    # for name, infoTuple in r.items():
    #     if(name=='currentKona'):
    #         i = infoTuple
    #     n = linearCombinationFromModel(m, infoTuple[0])
    #     lower = n[0]-2*np.sqrt(n[1])
    #     upper = n[0]+2*np.sqrt(n[1])
    #     sd = np.sqrt(n[1])
    #     mean = n[0]
    #
    #     print('{} = {} ({}) [{}, {}]'.format(name, mean[0], sd[0], lower[0], upper[0]))


    #form = getFormat('oldLeafVersa')

    # d = {'phevRange':40}

    # m = m[list(m.keys())[0]]
    # n = linearCombinationFromModel(m, r['currentS90'][0])

    print('Starting...')
    plotAllHeadToHeads()
    plotAllHeadToHeads('SUV')
    print('Finished!')
    print('Starting...')
    plotAllTimeslines()
    plotAllTimeslines('SUV')
    print('Finished!')