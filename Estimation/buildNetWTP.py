from helpFile import cleanNames
import pandas as p
import numpy as np
from tqdm import tqdm, trange
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from Inflator import Inflator
from scipy.stats import norm
from copy import deepcopy
from matplotlib.markers import MarkerStyle

def linearCombination(beta, varCov, r):
    beta = beta.reshape((-1,1))

    newBetas = r@beta
    newVarCov = r@varCov@np.transpose(r)

    return newBetas, newVarCov

def getParamIndices(paramNames, desiredParams, parSuffix = 'Mean'):
    r = []
    for param in desiredParams:
        if(param in paramNames):
            r.append(paramNames.index(param))
        elif((param+parSuffix) in paramNames):
            r.append(paramNames.index(param+parSuffix))

    return r

def getRelevantEstimates(modelData, parNames, parSuffix = 'Mean'):
    modelParNames = modelData['params']
    paramIndices = getParamIndices(modelParNames, parNames, parSuffix)

    rawBetas = modelData['coef']

    if('bestModel' in modelData.keys()):
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
        tempDelta = np.array(delta).reshape((-1,1))
        rawDeltas.append(tempDelta)

    r = np.hstack(tuple(rawDeltas))

    return r


def cleanParamDict(model, paramDict):
    modelParams = model['params']

    newParamDict = OrderedDict()

    for name, delta in paramDict.items():
        if(name in modelParams or (name+'Mean') in modelParams):
            newParamDict[name] = delta
        elif(name=='phevRange'):
            newParamDict['phev40'] = (delta-20)/(20)
            newParamDict['phev20'] = (40-delta) / (20)
        else:
            if(delta!=0):
                print('{} not found - dropping from calculation'.format(name))

    return newParamDict

def linearCombinationFromModel(model, paramDict):

    paramDict = cleanParamDict(model, paramDict)

    relBeta, relVarCov = getRelevantEstimates(model, paramDict.keys())
    rMatrix = createRMatrix(paramDict)

    r = linearCombination(relBeta, relVarCov, rMatrix)

    return r

def plotHeadToHeadStackedBar(model, paramDict, premiumPrice=None, title=None, saveFile = None, show=False, ylim=(-25, 13)):

    if (ylim == None):
        ylim = (-25, 13)

    x = 6
    gr = (1+np.sqrt(5))/2
    y = x/gr

    fig, ax = plt.subplots(1,1,figsize=(x,y))

    colors = sns.color_palette('colorblind', n_colors=10)

    netWTP, netWTPVar = linearCombinationFromModel(model, paramDict)
    netWTP, netWTPVar = float(netWTP), float(netWTPVar)
    netWTPSE = float(np.sqrt(netWTPVar))*2

    ax = plt.scatter(0, netWTP, marker='o', color='xkcd:red', s=50, zorder=4, label='Net WTP')
    ax = plt.errorbar(0, netWTP, yerr=netWTPSE, ecolor='xkcd:black', capsize=10, zorder=0, label='±2 S.E.', fmt='none')
    ax = plt.errorbar(0, netWTP, yerr=netWTPSE, ecolor='xkcd:black', capsize=30, zorder=1, fmt='none')

    xlim = plt.xlim()

    prevTempNetNeg = 0
    prevTempNetPos = 0
    i = 0
    print('lol-{}'.format(paramDict))
    for paramName, delta in paramDict.items():
        if (delta != 0):
            tempParamDict = {paramName: delta}
            tempNet, tempNetVar = linearCombinationFromModel(model, tempParamDict)
            tempNet, tempNetVar = float(tempNet), float(tempNetVar)
            tempCleanName = cleanNames(paramName)
            if('Capacity' in tempCleanName):
                print('lol-{}'.format(tempCleanName))
                tempCleanName = tempCleanName.replace(' ', '\n')

            if(tempNet<0):
                tempBottom = prevTempNetNeg
            else:
                tempBottom = prevTempNetPos

            if(paramName in ['oc', 'acc', 'tc', 'pc']):
                if(paramName=='oc'):
                    tempCleanName+='\n({} ¢/mile)'.format(np.round(delta,1))
                elif(paramName in ['pc', 'tc']):
                    tempCleanName+='\n({}k Lbs.)'.format(np.round(delta,1))
                else:
                    tempCleanName = 'Accel.'
                    tempCleanName+='\n({} sec.\n0-60 time)'.format(np.round(delta,1))
            elif (paramName == 'bevFC' or paramName == 'nobevFC'):
                tempCleanName = tempCleanName.replace('Fast', '\nFast')


            ax = plt.bar(0, tempNet, width=.005, bottom=tempBottom, label=tempCleanName, zorder=0, color=colors[i])
            if(tempNet<0):
                prevTempNetNeg+=tempNet
            else:
                prevTempNetPos+=tempNet
            i+=1

    plt.plot(xlim, [0,0], color='xkcd:black')
    if(premiumPrice!=None):
        plt.plot(xlim, [premiumPrice, premiumPrice], label='Price Premium', color='xkcd:cherry red')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks([])

    plt.legend(bbox_to_anchor=(1.01,1), loc='upper left')
    plt.ylabel('Willingness-to-Pay ($1,000s)')
    if (title != None):
        plt.title(title)
    if (show):
        plt.show()

    if (saveFile != None):
        plt.savefig('Plots/HeadToHeadStacked/' + saveFile, bbox_inches='tight', dpi=500)


def plotHeadToHeadWaterfall(model, paramDict, pricePremium=None, title=None, saveFile = None, show=False, ylim=None, addRebate = False, removeRebate = False, textRightAdjusted = False, aboveLine=True, splitAdjustment=False, direc='Plots', x = None, netText = None):

    if(x==None):
        x = 4.7
    gr = (1+np.sqrt(5))/2
    gr = 1
    y = x/gr

    fig, ax = plt.subplots(1,1,figsize=(x,y))
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    phevComparison = 'phevRange' in paramDict.keys()

    colors = sns.color_palette('deep', n_colors=10)
    # colors = sns.color_palette('Paired')

    woCreditColor = colors[0]
    woCreditColor = 'xkcd:red'

    # colors = sns.color_palette('bright', n_colors=10)
    wCreditColor = colors[9]
    wCreditColor = 'xkcd:violet'
    positiveColor = colors[2]
    negativeColor = colors[3]
    netColor = 'xkcd:blue'

    # woCreditColor = colors[9]
    # wCreditColor = colors[8]
    # positiveColor = colors[2]
    # negativeColor = colors[4]
    # netColor = colors[7]

    netWTP, netWTPVar = linearCombinationFromModel(model, paramDict)
    netWTP, netWTPVar = float(netWTP), float(netWTPVar)
    netWTPSE = float(np.sqrt(netWTPVar))*2


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
    noBEV = True
    for paramName, delta in paramDict.items():

        if('nobevFC' in paramDict.keys()):
            nobevFC = paramDict['nobevFC']
            if(nobevFC==0):
                bevFCStatement = ' With\nFast Charging'
            else:
                bevFCStatement = ' Without\nFast Charging'


        if(paramName=='hev' and delta!=0):
            relHEV=True

        if (paramName == 'bev' and delta != 0):
            noBEV = False

        if (delta != 0 or ('Range' in paramName and not phevComparison) or 'nobev' in paramName):
            tempParamDict = {paramName: delta}

            tempNet, tempNetVar = linearCombinationFromModel(model, tempParamDict)
            tempNet, tempNetVar = float(tempNet), float(tempNetVar)
            tempCleanName = cleanNames(paramName)
            tempBottom = prevTempNet

            if('Capacity' in tempCleanName or 'Cost' in tempCleanName):
                tempCleanName = tempCleanName.replace(' ', '\n')

            if (paramName in ['oc', 'acc', 'tc', 'pc']):
                if (paramName == 'oc'):
                    tempCleanName += '\n({} ¢/mi.)'.format(np.round(delta, 1))
                elif (paramName in ['pc', 'tc']):
                    tempCleanName += '\n({}k Lbs.)'.format(np.round(delta, 1))
                else:
                    tempCleanName = 'Accel.'
                    tempCleanName += '\n({} sec.\n0-60 time)'.format(np.round(delta, 1))
            elif(paramName=='bevFC' or paramName=='nobevFC'):
                tempCleanName = tempCleanName.replace('Fast','\nFast')
                tempCleanName = tempCleanName.replace('Charging', '\nCharging')
            elif(paramName=='asinhBEVRange'):
                tempCleanName = "BEV\n({} Mile\nRange{})".format(int(np.round(np.sinh(delta),0)),bevFCStatement)
            elif(paramName=='bevRange'):
                if(noBEV):
                    tempCleanName = 'HEV'
                else:
                    tempCleanName = 'BEV\n({} Mile\nRange{})'.format(int(delta) + 300, bevFCStatement)
            elif(paramName=='phevRange'):
                tempCleanName = 'PHEV\n({} Mile\nRange)'.format(int(delta))
            elif(paramName=='hev'):
                tempCleanName = 'HEV'
            if(paramName not in ['bev','hev']):
                newXTicks.append(i)
                newXLabels.append(tempCleanName)

            # ax = plt.bar(i, tempNet, width=width, bottom=tempBottom, label=tempCleanName, zorder=0, color=colors[i], edgecolor='xkcd:black')
            if(paramName not in ['bev','hev', 'nobevFC']):
                print('{} - Name'.format(paramName))

                if('Range' in paramName):
                    tempBottom=0
                    tempPlotNet = prevTempNet+tempNet
                else:
                    tempPlotNet = tempNet

                if(tempPlotNet<0):
                    color = negativeColor
                else:
                    color = positiveColor

                ax = plt.bar(i, tempPlotNet, width=width, bottom=tempBottom, zorder=0, color=color,
                             edgecolor='xkcd:black')


            prevTempNet += tempNet
            if (paramName not in ['bev','hev', 'nobevFC']):
                if (tempPlotNet < 0):
                    plt.text(i, prevTempNet-yShift, tempCleanName, va='top', ha='center')
                    # plt.text(i, prevTempNet - 1 - tempNet, tempCleanName, va='bottom', ha='center')
                else:
                    plt.text(i, prevTempNet+yShift, tempCleanName, va='bottom', ha='center')
                    # plt.text(i, prevTempNet + 1, tempCleanName, va='bottom', ha='center', color='xkcd:green', fontweight=0)
                    # plt.text(i, prevTempNet - 1 - tempNet, tempCleanName, va='top', ha='center')

            i+=1
            if(paramName not in ['bev','hev', 'nobevFC']):
                if (prevTempNet < minTempNet):
                    minTempNet = prevTempNet
                plt.plot([i-1, i], [prevTempNet, prevTempNet], color='xkcd:black')

    # ax = plt.bar(i, netWTP, color='xkcd:red', zorder=4, label='Net WTP', width=width)
    # ax = plt.scatter(i, netWTP, marker='o', color='xkcd:red', s=50, zorder=4, label='Net WTP')
    ax = plt.scatter(i, netWTP, marker='o', color=netColor, s=50, zorder=4, edgecolors=netColor)
    ax = plt.errorbar(i, netWTP, yerr=netWTPSE, ecolor=netColor, capsize=10, zorder=1, label='±2 S.E.', fmt='none')
    ax = plt.text(i, netWTP+netWTPSE+yShift, "Net", ha='center', va='bottom')
    ax = plt.text(i, netWTP - netWTPSE - yShift, netText, ha='center', va='top')
    # ax = plt.errorbar(i, netWTP, yerr=netWTPSE, ecolor='xkcd:black', capsize=20, zorder=1, fmt='none')
    newXTicks.append(i)
    newXLabels.append('Net')
    xlim = plt.xlim()
    xlim = list(xlim)
    xlim[1]+=.1
    tempYlim = plt.ylim()
    print('xlim: {}'.format(xlim))


    plt.plot(xlim, [0,0], color = 'xkcd:light grey', alpha=1, zorder=-5)
    # plt.fill_between(xlim, [0, 0], [min(ylim), min(ylim)], color = 'xkcd:light grey', alpha=.75, zorder=-5)


    if(textRightAdjusted):
        textXPlacement = max(xlim)
        haArg = 'right'
    else:
        textXPlacement = min(xlim)
        haArg = 'left'
    if(aboveLine):
        vaArg = 'bottom'
        yShift = 0
    else:
        vaArg = 'top'
        yShift = -0.75

    if(splitAdjustment):
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

    if(phevComparison):
        powertrainType = 'PHEV'
    else:
        powertrainType = 'BEV'

    if(pricePremium!=None):

        if(not (addRebate or removeRebate)):
            plt.plot(xlim, [pricePremium, pricePremium], label='{} Price\nPremium'.format(powertrainType), color=woCreditColor, alpha=alphaVal, zorder=zOrderVal)
            plt.text(textXPlacement, pricePremium+yShift, "{} Price\nPremium".format(powertrainType), ha=haArg, va=vaArg, alpha=alphaVal, zorder=zOrderVal, color=woCreditColor)
        else:
            if(addRebate):
                plt.plot(xlim, [pricePremium, pricePremium], label='{} Price\nPremium'.format(powertrainType), color=woCreditColor, alpha=alphaVal, zorder=zOrderVal)
                plt.plot(xlim, [pricePremium-7.5, pricePremium-7.5], label='{} Price Premium\nwith $7,500 Credit'.format(powertrainType),
                         color=wCreditColor, alpha=alphaVal, zorder=zOrderVal)
                if(splitAdjustment):
                    print('lol-it works-{}-{}-{}-{}'.format(altTextXPlacement, altYShift, altHaArg, altVaArg))
                    plt.text(altTextXPlacement, pricePremium + altYShift, "{} Price\nPremium".format(powertrainType), ha=altHaArg, va=altVaArg,
                             color=woCreditColor, alpha=alphaVal, zorder=zOrderVal)
                else:
                    plt.text(textXPlacement, pricePremium + yShift, "{} Price\nPremium".format(powertrainType), ha=haArg, va=vaArg,
                             color=woCreditColor, alpha=alphaVal, zorder=zOrderVal)
                plt.text(textXPlacement, pricePremium-7.5+yShift, "{} Price Premium\nwith $7,500 Credit".format(powertrainType), ha=haArg, va=vaArg, color=wCreditColor, alpha=alphaVal, zorder=zOrderVal)

            if(removeRebate):
                if (splitAdjustment):
                    plt.text(altTextXPlacement, pricePremium + altYShift, "{} Price\nPremium".format(powertrainType), ha=altHaArg, va=altVaArg,
                             color=woCreditColor, alpha=alphaVal, zorder=zOrderVal)
                else:
                    plt.text(textXPlacement, pricePremium + yShift, "{} Price\nPremium".format(powertrainType), ha=haArg, va=vaArg,
                             color=woCreditColor, alpha=alphaVal, zorder=zOrderVal)
                plt.plot(xlim, [pricePremium, pricePremium], label='{} Price Premium\nwith $7,500 Credit'.format(powertrainType),
                         color=wCreditColor, alpha=alphaVal, zorder=zOrderVal)
                plt.text(textXPlacement, pricePremium+yShift, "{} Price Premium\nwith $7,500 Credit".format(powertrainType), ha=haArg, va=vaArg, color=wCreditColor, alpha=alphaVal, zorder=zOrderVal)
                plt.text(textXPlacement, pricePremium+7.5+yShift, "{} Price\nPremium".format(powertrainType), ha=haArg, va=vaArg, color=woCreditColor, alpha=alphaVal, zorder=zOrderVal)

    plt.xlim(xlim)

    plt.xticks([])
    # plt.xticks(newXTicks, newXLabels)

    plt.ylim(ylim)

    if(minTempNet< min(plt.ylim()) and ylim==None):
        plt.ylim(bottom=minTempNet - 1)

    # plt.legend(bbox_to_anchor=(1.01,1), loc='upper left')
    if(relHEV):
        if(noBEV):
            plt.ylabel('HEV WTP Relative to ICEV'.format(powertrainType))
        else:
            plt.ylabel('{} WTP Relative to HEV'.format(powertrainType))
    else:
        plt.ylabel('{} WTP Relative to CV'.format(powertrainType))


    if('currentLeaf' in saveFile or 'futureLeaf' in saveFile):
        plt.ylabel('')
        yticks = plt.yticks()
        print(yticks)
        plt.yticks(yticks[0], [])
    else:
        newYTicks = []
        newYLabs = []
        for ytick in plt.yticks()[0]:
            newYTicks.append(int(ytick))
            newYLabs.append('${}k'.format(int(ytick)))
        plt.yticks(newYTicks, newYLabs)

    if (title != None):
        plt.title(title)
    if (show):
        plt.show()

    if (saveFile != None):
        plt.savefig('{}/HeadToHeadWaterfall/'.format(direc) + saveFile, bbox_inches='tight', dpi=300)
    else:
        plt.show()



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
    r.loc[:, 'Err'] = 2*r.loc[:, 'Std. Err.']
    r.loc[:, 'Lower'] = r.loc[:, 'Mean']-2*r.loc[:, 'Std. Err.']
    r.loc[:, 'Upper'] = r.loc[:, 'Mean'] + 2 * r.loc[:, 'Std. Err.']
    r.loc[:, 'Premium'] = r.loc[:, 'Premium']/1000
    return r

def plotTimeline(timeDict, model, title=None, saveFile = None):
    df = constructTimelineDF(timeDict, model)
    df = df.sort_values('Year')



    x = 2
    gr = (1+np.sqrt(5))/2
    gr = 1
    y = x*gr

    netWTPColor = 'xkcd:blue'
    errorColor = 'xkcd:light blue'
    premiumColor = 'xkcd:red'


    fig, ax = plt.subplots(1,1,figsize=(x,y))
    #ax = sns.lineplot(x='Year', y='Mean', data=df, color=netWTPColor, ls='--')
    ax = sns.lineplot(x='Year', y='Premium', data=df, color=premiumColor, ls='--')

    ax = plt.errorbar(x='Year', y='Mean', yerr='Err', color=netWTPColor, data=df,
                      capsize=10, ls='--', label=None)

    for i in range(2):
        if(i == 0):
            ax = sns.scatterplot(x='Year', y='Premium', data=df.loc[df.loc[:, 'Year']<2023, :], s=50, color=premiumColor)
            ax = sns.scatterplot(x='Year', y='Mean', color=netWTPColor, data=df.loc[df.loc[:, 'Year']<2023, :], s=50)
        else:
            ax = sns.scatterplot(x='Year', y='Premium', color="xkcd:white", data=df.loc[df.loc[:, 'Year']>2023, :], s=50, edgecolor=premiumColor)
            ax = sns.scatterplot(x='Year', y='Mean', color='xkcd:white', data=df.loc[df.loc[:, 'Year']>2023, :], s=50, edgecolor=netWTPColor)

    plt.ylim([-20, 25])
    ylim = plt.ylim()
    xlim = plt.xlim()

    plt.ylim(ylim)
    plt.xlim([2011, max(df.loc[:, 'Year'])+1])
    ylim = plt.ylim()
    xlim = plt.xlim()

    if(min(df.loc[:, 'Year'])<2018):
        x, y = max(xlim) - .01 * (max(xlim) - min(xlim)), max(ylim)-.0*(max(ylim)-min(ylim))
        ha = 'right'
    else:
        x, y = min(xlim) + .08 * (max(xlim) - min(xlim)), max(ylim) - .1 * (max(ylim) - min(ylim))
        ha = 'left'

    plt.text(x,y, 'BEV\nPrice\nPremium', va='top', ha=ha, color=premiumColor)


    if (min(df.loc[:, 'Year'])<2018):
        x, y = max(xlim) - .01 * (max(xlim) - min(xlim)), max(ylim) - .73 * (max(ylim) - min(ylim))
    else:
        x, y = min(xlim) + .05 * (max(xlim) - min(xlim)), max(ylim) - .6 * (max(ylim) - min(ylim))
    #plt.fill_between([2022, 2030], y1=[max(ylim)]*2, y2=[min(ylim)]*2, color='xkcd:light gray', zorder=0)

    plt.text(x, y, 'BEV\nNet WTP', va='top', ha=ha, color=netWTPColor)

    if(title!=None):
        plt.title(title)

    newXTicks = set(df.loc[:, 'Year'])
    newXTicks.add(2013)
    plt.xticks(list(newXTicks))
    #plt.ylabel('$1,000s')
    plt.ylabel('')
    yticks, ylabs = plt.yticks()

    newLabs = []
    newTicks = []

    for ytick in yticks:

        newYTick = int(ytick)
        newTicks.append(newYTick)
        if(ytick==0):
            newLabs.append('$0')
        elif(ytick>0):
            newLabs.append('${}k'.format(newYTick))
        else:
            newLabs.append('-${}k'.format(abs(newYTick)))


    plt.ylim([-22, 25])
    xlim = plt.xlim()
    ax = plt.plot(xlim, [0,0], color='xkcd:gray', zorder=-5)
    plt.xlim(xlim)
    if ('leaf' in saveFile or 'kona' in saveFile or 'prius' in saveFile or 'outlander' in saveFile or 'wrangler' in saveFile or 'aviator' in saveFile): # in saveFile or 'PHEV' in saveFile or 'xe' in saveFile or 'prime' in saveFile or 'a7' in saveFile or 's60' in saveFile or 's90' in saveFile
        plt.yticks(newTicks, newLabs)
        print(saveFile)
    else:
        plt.yticks(newTicks, [])
    if(saveFile!=None):
        plt.savefig('Plots/TimelinePlots/{}.png'.format(saveFile), dpi=300, bbox_inches='tight')
    # plt.show()

    return df

