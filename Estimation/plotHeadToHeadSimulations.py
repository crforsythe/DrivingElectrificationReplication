import pandas as p
import numpy as np
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from buildPlots import getFullData


def getSpecData():
    fullData = getFullData()
    r = OrderedDict()
    for vehType in ['Car', 'SUV']:
        df = fullData[vehType]
        print(df)
        groups = list(set(df.loc[:, 'groupName']))
        r[vehType] = OrderedDict()
        for group in groups:
            groupData = df.loc[df.loc[:, 'groupName']==group, :]
            models = list(groupData.loc[:, 'model'])
            modelsPretty = list(groupData.loc[:, 'prettyModel'])
            r[vehType][group] = OrderedDict()
            r[vehType][group]['priceDiff'] = (max(groupData.loc[:, 'price'])-min(groupData.loc[:, 'price']))/1000
            r[vehType][group]['Year'] = np.mean(groupData.loc[:, 'year'])
            r[vehType][group]['models'] = '{}-{}'.format(models[0], models[1])
            r[vehType][group]['cvModel'] = modelsPretty[1]
            r[vehType][group]['evModel'] = modelsPretty[0]
        r[vehType] = p.DataFrame.from_dict(r[vehType])
        r[vehType] = r[vehType].transpose()
        r[vehType].loc[:, 'group'] = r[vehType].index


    return r


def plotTimelineBar(df, saveFile, width=5):
    df.loc[:, 'Year'] = df.loc[:, 'Year'].astype(int)
    df = df.sort_values('Year')
    print(df)

    x = 1.5
    gr = (1 + np.sqrt(5)) / 2
    y = x * gr

    netWTPColor = 'xkcd:light green'
    errorColor = 'xkcd:blue'
    premiumColor = 'xkcd:light blue'

    fig, ax = plt.subplots(1, 1, figsize=(x, y))
    print(list(df.columns))
    errArg = np.transpose(
        np.hstack((np.array(-df.loc[:, '2.5Diff']).reshape((-1, 1)), np.array(df.loc[:, '97.5Diff']).reshape((-1, 1)))))

    ax.errorbar(x=df.loc[:, 'Year'], y=df.loc[:, 'mean'], yerr=errArg, color=errorColor,
                capsize=7, ls='', label=None)
    df.loc[:, 'invMean'] = 1-df.loc[:, 'mean']
    for i in range(2):
        if (i == 0):
            ax = plt.bar(x='Year', height='mean', color=netWTPColor, data=df.loc[df.loc[:, 'Year'] < 2023, :], width=width)
            ax = plt.bar(x='Year', height='invMean', bottom='mean', color=premiumColor, data=df.loc[df.loc[:, 'Year'] < 2023, :],
                         width=width)
        else:
            # ax = plt.bar(x='Year', height='mean', color='xkcd:white', data=df.loc[df.loc[:, 'Year'] > 2023, :], width=width,
            #              edgecolor=netWTPColor)
            yVal = np.mean(df.loc[df.loc[:, 'Year'] > 2023, 'mean'])
            ax = plt.fill_between(x=[2030-width/2, 2030+width/2], y1=[yVal, yVal], y2=[0, 0], facecolor=netWTPColor,
                         hatch='/')
            ax = plt.fill_between(x=[2030 - width / 2, 2030 + width / 2], y1=[1, 1], y2=[yVal, yVal],
                                  facecolor=premiumColor,
                                  hatch='/')
    #
    ylim = plt.ylim()

    xlim = plt.xlim()
    plt.xlim([2011, max(xlim)])
    newXTicks = set(df.loc[:, 'Year'])
    newXTicks.add(2013)
    print(newXTicks)
    plt.xticks(list(newXTicks))

    maxYear = max(df.loc[:, 'Year'])
    for ind, row in df.iterrows():
        ax = plt.plot([row['Year']-width/2, row['Year']-width/2], [0,1], color='xkcd:black')
        ax = plt.plot([row['Year'] + width / 2, row['Year'] + width / 2], [0, 1], color='xkcd:black')
        if(row['Year']==maxYear):
            finalYVal = row['mean']
            cvModel = row['cvModel']
            evModel = row['evModel']


    plt.ylim([0, 1])
    # premAx.set_ylim([-25, 25])
    xlim = plt.xlim()
    plt.xlim(xlim)

    yticks, ylabels = plt.yticks()

    newYTicks = []
    newYLabels = []

    for ytick in yticks:
        newYTicks.append(int(ytick * 100) / 100)
        newYLabels.append('{}%'.format(int(ytick * 100)))

    if('leaf' in saveFile or 'kona' in saveFile or 'prius' in saveFile or 'outlander' in saveFile or 'wrangler' in saveFile or 'aviator' in saveFile):
        plt.yticks(newYTicks, newYLabels)
    else:
        plt.yticks(newYTicks, [])

    xlim = plt.xlim()

    plt.text(max(xlim)+(max(xlim)-min(xlim))*.03, finalYVal, 'vs.')


    if (' ' in cvModel):
        cvModel = cvModel.split(' ')
        plt.text(max(xlim) + (max(xlim) - min(xlim)) * .03, finalYVal + .07, cvModel[1])
        plt.text(max(xlim) + (max(xlim) - min(xlim)) * .03, finalYVal + .14, cvModel[0])
    elif ('-' in cvModel):
        cvModel = cvModel.split('-')
        plt.text(max(xlim) + (max(xlim) - min(xlim)) * .03, finalYVal + .07, cvModel[1])
        plt.text(max(xlim) + (max(xlim) - min(xlim)) * .03, finalYVal + .14, cvModel[0])
    else:
        plt.text(max(xlim) + (max(xlim) - min(xlim)) * .03, finalYVal + .07, cvModel)

    if(' ' in evModel):
        evModel = evModel.split(' ')
        plt.text(max(xlim) + (max(xlim) - min(xlim)) * .03, finalYVal - .07, evModel[0], color='xkcd:green')
        plt.text(max(xlim) + (max(xlim) - min(xlim)) * .03, finalYVal - .14, evModel[1], color='xkcd:green')
    else:
        plt.text(max(xlim) + (max(xlim) - min(xlim)) * .03, finalYVal - .07, evModel, color='xkcd:green')

    if (saveFile != None):
        plt.savefig('Plots/HeadToHeadSimPlots/{}.png'.format(saveFile), dpi=300, bbox_inches='tight')
    # plt.show()



def plotTimeline(df, saveFile):
    df.loc[:, 'Year'] = df.loc[:, 'Year'].astype(int)
    df = df.sort_values('Year')

    x = 2
    gr = (1 + np.sqrt(5)) / 2
    y = x * gr

    netWTPColor = 'xkcd:blue'
    errorColor = 'xkcd:light blue'
    premiumColor = 'xkcd:red'

    fig, ax = plt.subplots(1, 1, figsize=(x, y))
    premAx = ax.twinx()
    # ax = sns.lineplot(x='Year', y='Mean', data=df, color=netWTPColor, ls='--')
    sns.lineplot(x='Year', y='priceDiff', data=df, color=premiumColor, ls='--', ax=premAx)

    errArg = np.transpose(np.hstack((np.array(-df.loc[:, '2.5Diff']).reshape((-1,1)), np.array(df.loc[:, '97.5Diff']).reshape((-1,1)))))

    ax.errorbar(x=df.loc[:, 'Year'], y=df.loc[:,'mean'], yerr=errArg, color=netWTPColor,
                      capsize=10, ls='--', label=None)

    for i in range(2):
        if (i == 0):
            ax = plt.bar(x='Year', y='mean', color=netWTPColor, data=df.loc[df.loc[:, 'Year'] < 2023, :], ax=ax)
        else:
            ax = plt.bar(x='Year', y='mean', color='xkcd:white', data=df.loc[df.loc[:, 'Year'] > 2023, :], edgecolor=netWTPColor, ax=ax)
    #
    ylim = plt.ylim()

    xlim = plt.xlim()

    newXTicks = set(df.loc[:, 'Year'])
    newXTicks.add(2013)
    if('leaf' in saveFile or 'kona' in saveFile):
        plt.xticks(newXTicks)
        print('{}-lol'.format(saveFile))
    else:
        print('{}-lol2'.format(saveFile))
        plt.xticks(newXTicks, None)

    plt.ylim([0,1])
    # premAx.set_ylim([-25, 25])
    xlim = plt.xlim()
    plt.xlim(xlim)




    if (saveFile != None):
        plt.savefig('Plots/HeadToHeadSimPlots/{}.png'.format(saveFile), dpi=300, bbox_inches='tight')
    # plt.show()

def plotTimelines(fullData):
    for vehType, df in fullData.items():
        print(vehType)
        models = set(df.loc[:, 'models'])
        for model in models:
            tempDF = df.loc[df.loc[:, 'models']==model, :]
            savefile = model
            print(savefile)

            plotTimelineBar(tempDF, savefile)





fileDict = {'Car':'HeadToHeadSims/pooled-car-linear-mixed-weight2018.csv', 'SUV':'HeadToHeadSims/pooled-suv-linear-mixed-weight2018.csv'}


r = OrderedDict()
for vehType, file in fileDict.items():
    data = p.read_csv(file)
    r[vehType] = OrderedDict()
    for col in data.columns:

        tempR = {'mean':np.mean(data.loc[:, col]), '2.5':np.percentile(data.loc[:, col], 2.5), '97.5':np.percentile(data.loc[:, col], 97.5)}
        tempR['2.5Diff'] = tempR['2.5']-tempR['mean']
        tempR['97.5Diff'] = tempR['97.5'] - tempR['mean']
        r[vehType][col] = tempR

    r[vehType] = p.DataFrame.from_dict(r[vehType])
    r[vehType] = r[vehType].transpose()
    r[vehType].loc[:, 'group'] = r[vehType].index

fullData = getFullData()
fullData = getSpecData()

for vehType in fullData.keys():
    fullData[vehType] = fullData[vehType].merge(r[vehType], on='group')

plotTimelines(fullData)