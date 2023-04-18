# import pylab as p
import numpy as np
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as p
from helpFile import linearCombinationFromModel, plotBEVCoefficents, plotCoefficients, plotMultipleWTPOverRange
from collections import OrderedDict, Counter
from tqdm import tqdm
from scipy.stats import norm
from getPythonModels import *


types = ['car', 'suv']
weights = ['unweighted']
funcForms = ['linear']
modelSpecs = ['mixed', 'simple']
i = 0

startColorDict = dict(zip(types, [0,3]))
ylimDict = dict(zip(types, [(-8, 10), (-12, 15)]))

#Build Plots for covid unaffected group
for vehType in types:
    for weight in weights:
        for form in funcForms:
            for model in modelSpecs:
                disc = ['-{}'.format(vehType), '-{}'.format(weight), '-{}'.format(form), '-{}'.format(model)]
                pooledModel = loadPooledUnweightedModels(disc, addMeanName=True)
                covidPooledModels = loadPooledCovidModels(disc, addMeanName=True)


                models = joinDicts(pooledModel, covidPooledModels)


                saveFile = 'studyComparison-pooled-{}-{}-{}-{}'.format(vehType, weight, form, model)
                if(vehType=='suv'):
                    cleanType = vehType.upper()
                else:
                    cleanType = vehType.capitalize()
                print(saveFile)
                title = 'BEV Willingness to Pay - {} Samples \n {} {} Logit Model Means'.format(cleanType, weight.capitalize(), model.capitalize())
                title = None
                phevTitle = None
                designTitle = None
                brandTitle = None
                if(title!=None):
                    phevTitle = title.replace('BEV', '(P)HEV')
                    designTitle = title.replace('BEV', 'Design')
                    brandTitle = title.replace('BEV', 'Brand')
                if(title!=None):
                    title += '\n Weighted to Same Baseline'
                    phevTitle += '\n Weighted to Same Baseline'
                    designTitle += '\n Weighted to Same Baseline'
                    brandTitle += '\n Weighted to Same Baseline'
                names = ['2021 Study Data \n Covid Unaffected', '2021 Study Data']
                names.reverse()
                tempStartColor = startColorDict[vehType]
                plotMultipleWTPOverRange(models, names, title=title, saveFile=saveFile, startColors=tempStartColor,
                                         direc='Plots/PlotsCovidUnaffected/BEV Range Plots Parameterized')
                r = plotCoefficients(models, names, ['hev', 'phev20', 'phev40'], title=phevTitle,
                                     saveFile='HEV Plots/' + saveFile, startColors=tempStartColor,
                                     ylim=ylimDict[vehType], saveDirec='Plots/PlotsCovidUnaffected')
                rDesign = plotCoefficients(models, names, ['oc', 'acc', 'nobevFC'], title=designTitle,
                                           saveFile='Design Plots/' + saveFile, startColors=tempStartColor,
                                           ylim=ylimDict[vehType], saveDirec='Plots/PlotsCovidUnaffected')
                r = plotCoefficients(models, names, ['american', 'skorean', 'japanese', 'chinese'], title=brandTitle,
                                     saveFile='Brand Plots/' + saveFile, startColors=tempStartColor,
                                     ylim=(-25, max(ylimDict[vehType])), saveDirec='Plots/PlotsCovidUnaffected')