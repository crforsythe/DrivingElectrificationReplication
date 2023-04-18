import pandas as p
import numpy as np
from MarketSimulation import loadASCs, predictAllNewVehicles, loadMarketData
from collections import OrderedDict
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns
def getBEVProportion(data, today=False):
    if(today):
        subData = data.loc[data.loc[:, 'Original']==1, :]
        return np.sum(subData.loc[subData.loc[:, 'type'] == 'BEV', 'marketShare'])
    else:
        return np.sum(data.loc[data.loc[:, 'type'] == 'BEV', 'predictedMeanShare'])

def getPEVProportion(data, today=False):
    if(today):
        subData = data.loc[data.loc[:, 'Original'] == 1, :]
        return np.sum(subData.loc[subData.loc[:, 'type'].isin(['BEV', 'PHEV']), 'marketShare'])
    else:
        return np.sum(data.loc[data.loc[:, 'type'].isin(['BEV', 'PHEV']), 'predictedMeanShare'])

def getProportionByType():
    types = ['Car', 'SUV', 'Pickup']
    reverseTypes = ['car', 'suv', 'truck']

    typeMapping = dict(zip(types, reverseTypes))

    r = OrderedDict()
    for vehType in types:
        r[typeMapping[vehType]] = np.sum(loadMarketData(vehType).loc[:, 'Sales'])

    totalSum = np.sum(list(r.values()))

    for vehType in types:
        r[typeMapping[vehType]] = r[typeMapping[vehType]]/totalSum

    return r

def constructPropDF(bevPropDict, pevPropDict, bevTodayProps, pevTodayProps):
    scenCol = []
    vehTypeCol = []
    modelTypeCol = []

    bevPropCol = []
    pevPropCol = []
    bevTodayPropCol = []
    pevTodayPropCol = []

    for scen, vehTypeDict in bevPropDict.items():
        for vehType, modelTypeDict in vehTypeDict.items():
            for modelType, propDict in modelTypeDict.items():
                scenCol.append(scen)
                vehTypeCol.append(vehType)
                modelTypeCol.append(modelType)

                bevPropCol.append(bevPropDict[scen][vehType][modelType])
                pevPropCol.append(pevPropDict[scen][vehType][modelType])

                bevTodayPropCol.append(bevTodayProps[scen][vehType][modelType])
                pevTodayPropCol.append(pevTodayProps[scen][vehType][modelType])

    scenCol = np.array(scenCol).reshape((-1,1))
    vehTypeCol = np.array(vehTypeCol).reshape((-1,1))
    modelTypeCol = np.array(modelTypeCol).reshape((-1,1))
    bevPropCol = np.array(bevPropCol).reshape((-1,1))
    pevPropCol = np.array(pevPropCol).reshape((-1,1))
    bevTodayPropCol = np.array(bevTodayPropCol).reshape((-1, 1))
    pevTodayPropCol = np.array(pevTodayPropCol).reshape((-1, 1))

    cols = ['Scenario', 'VehType', 'Model Type', 'Proportion BEV', 'Proportion PEV', 'Proportion BEV Today', 'Proportion PEV Today']

    df = p.DataFrame(np.hstack((scenCol, vehTypeCol, modelTypeCol, bevPropCol, pevPropCol, bevTodayPropCol, pevTodayPropCol)), columns=cols)

    df.loc[:, 'Proportion BEV'] = df.loc[:, 'Proportion BEV'].astype(float)
    df.loc[:, 'Proportion PEV'] = df.loc[:, 'Proportion PEV'].astype(float)
    df.loc[:, 'Proportion BEV Today'] = df.loc[:, 'Proportion BEV Today'].astype(float)
    df.loc[:, 'Proportion PEV Today'] = df.loc[:, 'Proportion PEV Today'].astype(float)

    return df

def plotProps(propDF, width=.8, bev=True, plotSuffix=None):

    x = 4
    gr = (1+np.sqrt(5))/2
    y = x/gr

    scenarios = []

    vehTypeDict = {'car':'Car', 'suv':'SUV', 'truck':'Pickup'}
    vehColorDict = {'car':'xkcd:blue', 'suv':'xkcd:red', 'truck':'xkcd:green'}

    for ind, row in propDF.iterrows():
        if(row['Scenario'] not in scenarios):
            scenarios.append(row['Scenario'])



    for vehType in list(set(propDF.loc[:, 'VehType'])):
        for modelType in list(set(propDF.loc[:, 'Model Type'])):
                fig, ax = plt.subplots(1, 1, figsize=(x, y))
                subData = propDF.loc[(propDF.loc[:, 'Model Type']==modelType)&(propDF.loc[:, 'VehType']==vehType)]

                for ind, row in subData.iterrows():
                    if(bev):
                        plotProp = row['Proportion BEV']
                        todayProp = row['Proportion BEV Today']
                        plotText = 'BEV'
                    else:
                        plotProp = row['Proportion PEV']
                        todayProp = row['Proportion PEV Today']
                        plotText = 'PEV'
                    ax = plt.bar(scenarios.index(row['Scenario']), plotProp, width=width, color=vehColorDict[vehType], alpha=.5)
                    ax = plt.text(scenarios.index(row['Scenario']), plotProp/2, plotText, ha='center', va='center')
                    ax = plt.text(scenarios.index(row['Scenario']), plotProp+((1-plotProp)/2), 'Other', ha='center', va='center')


                plt.xticks(range(len(scenarios)), scenarios)
                xlim = plt.xlim()

                plt.plot(xlim, [todayProp, todayProp], '--', color='xkcd:black')
                plt.xlim(xlim)

                if(bev):

                    if(vehType=='car'):
                        plt.ylabel('BEV Market Share'.format(vehTypeDict[vehType]))
                    plt.ylabel('BEV Market Share'.format(vehTypeDict[vehType]))
                    plt.title('US BEV {} Market Share'.format(vehTypeDict[vehType]))
                    plt.text(max(xlim)*1.03, todayProp, 'Real\nMY2020\nBEV Share')
                else:
                    plt.ylabel('US PEV {} Market Share'.format(vehTypeDict[vehType]))
                    plt.text(max(xlim)*1.03, todayProp, 'Real\nMY2020\nPEV Share')


                plt.ylim([0,1])

                ylim = plt.ylim()

                yticks, ylabels = plt.yticks()

                newYTicks = []
                newYLabels = []

                for ytick in yticks:
                    newYTicks.append(int(ytick * 100) / 100)
                    newYLabels.append('{}%'.format(int(ytick * 100)))

                if(vehType=='car'):
                    plt.yticks(newYTicks, newYLabels)
                else:
                    plt.yticks(newYTicks, ['']*len(newYTicks))
                    plt.yticks(newYTicks, newYLabels)



                for i in range(len(scenarios)):
                    plt.plot([i - width / 2, i - width / 2], ylim, color='xkcd:black')
                    plt.plot([i + width / 2, i + width / 2], ylim, color='xkcd:black')
                plt.ylim(ylim)
                if(plotSuffix==None):
                    plt.savefig('Plots/MarketSim/{}-{}.png'.format(vehType, modelType), bbox_inches='tight')
                else:
                    plt.savefig('Plots/MarketSim/{}-{}-{}.png'.format(vehType, modelType, plotSuffix), bbox_inches='tight')


def applyOCs(ascData, newFuelPrices=False):
    if (newFuelPrices):
        gasPriceNew = 427.1
        elecCostNew = 13.83

        gasPriceOld = 263.6
        elecCostOld = 13.04
    else:
        gasPriceNew = 263.6
        elecCostNew = 13.04

        gasPriceOld = 263.6
        elecCostOld = 13.04

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

def runSimulations(newFuelPrices=False, accelerationProp=1, plotSuffix=None, altOCPropValues=False, loadSuffix = '', refined=True):

    scenarios = {
        'MY2020': {'bevRange': 200, 'pricePremium': 1.4828},
        'MY2030': {'bevRange': 300, 'pricePremium': 1}}



    r = OrderedDict()
    bevProps = OrderedDict()
    pevProps = OrderedDict()
    bevTodayProps = OrderedDict()
    pevTodayProps = OrderedDict()
    ascData = loadASCs(refined=refined, suffix=loadSuffix)

    for scenTitle, scenSpec in scenarios.items():
        print('-'*100)
        print(scenTitle)

        if(newFuelPrices):
            r[scenTitle] = predictAllNewVehicles(ascData, newRange=scenSpec['bevRange'],
                                                 newPropPricePremium=scenSpec['pricePremium'], newFuelPrices=newFuelPrices, acclerationProp=accelerationProp, altOCPropValues=altOCPropValues)
        else:
            r[scenTitle] = predictAllNewVehicles(ascData, newRange=scenSpec['bevRange'],
                                                 newPropPricePremium=scenSpec['pricePremium'], acclerationProp=accelerationProp, altOCPropValues=altOCPropValues)
        bevProps[scenTitle] = OrderedDict()
        pevProps[scenTitle] = OrderedDict()
        bevTodayProps[scenTitle] = OrderedDict()
        pevTodayProps[scenTitle] = OrderedDict()
        for vehType, modelDict in r[scenTitle].items():
            bevProps[scenTitle][vehType] = OrderedDict()
            pevProps[scenTitle][vehType] = OrderedDict()
            bevTodayProps[scenTitle][vehType] = OrderedDict()
            pevTodayProps[scenTitle][vehType] = OrderedDict()
            for modelType, marketData in modelDict.items():
                bevProps[scenTitle][vehType][modelType] = getBEVProportion(r[scenTitle][vehType][modelType])
                pevProps[scenTitle][vehType][modelType] = getPEVProportion(r[scenTitle][vehType][modelType])
                bevTodayProps[scenTitle][vehType][modelType] = getBEVProportion(r[scenTitle][vehType][modelType], today=True)
                pevTodayProps[scenTitle][vehType][modelType] = getPEVProportion(r[scenTitle][vehType][modelType], today=True)
        print('-' * 100)

    propDF = constructPropDF(bevProps, pevProps, bevTodayProps, pevTodayProps)
    plotProps(propDF, plotSuffix=plotSuffix)

    return propDF

r = runSimulations(refined=True)
rLowOCWTP = runSimulations(refined=True, loadSuffix='-LowAltOC', plotSuffix='LowAltOC')
rHighOCWTP = runSimulations(refined=True, loadSuffix='-HighAltOC', plotSuffix='HighAltOC')
rAltAcc = runSimulations(refined=True, accelerationProp=1-.2556, plotSuffix='optimisticAcc')
rAltFuelPrices = runSimulations(refined=True, newFuelPrices=True, plotSuffix='altFuelPrices')