import pandas as p
import numpy as np
from tqdm import tqdm
import stata_setup as s
import os
import sys
if(sys.platform=='darwin'):
    s.config('/Applications/Stata', 'mp')
else:
    s.config("C:\Program Files\Stata17", 'mp')
from pystata import stata
from collections import OrderedDict
from Inflator import Inflator
from copy import deepcopy
from scipy.optimize import minimize, Bounds, LinearConstraint
from datetime import datetime
from multiprocessing import cpu_count, Pool
from tqdm import tqdm

def getDynataPilotDemoData(type='all'):

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/DynataTest/DynataPilot.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\DynataTest\DynataPilot.xlsx'

    type = type.lower()
    possibleTypes = ['car', 'suv', 'truck', 'all']
    if(type not in possibleTypes):
        print('Type not available.')
    else:
        sheetPrefix = 'weight-'
        if(type!='all'):
            sheet = sheetPrefix+type
            data = p.read_excel(file, sheet)
            data.loc[:, 'Vehicle Type'] = type
            data = appendCategorical(data)
            return data
        else:
            datas = []
            for tempType in possibleTypes[:len(possibleTypes)-1]:
                tempData = getDynataPilotDemoData(tempType)
                datas.append(tempData)
            r = datas[0]

            for data in datas[1:]:
                r = r.append(data)
            return r
def getDynataDemoData(type='all'):

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Dynata/Dynata.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Dynata\Dynata.xlsx'

    type = type.lower()
    possibleTypes = ['car', 'suv', 'truck', 'all']
    if(type not in possibleTypes):
        print('Type not available.')
    else:
        sheetPrefix = 'weight-'
        if(type!='all'):
            sheet = sheetPrefix+type
            data = p.read_excel(file, sheet)
            data.loc[:, 'Vehicle Type'] = type

            data.loc[data.loc[:, 'householdSize'] == '7 or more', 'householdSize'] = 7
            data = data.replace('I do not wish to answer', np.nan)
            data.loc[:, 'householdSize'] = data.loc[:, 'householdSize'].astype(float)

            data = appendCategorical(data)
            return data
        else:
            datas = []
            for tempType in possibleTypes[:len(possibleTypes)-1]:
                tempData = getDynataDemoData(tempType)
                datas.append(tempData)
            r = datas[0]

            for data in datas[1:]:
                r = r.append(data)
            return r

def getPooledData(type = 'all'):

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Pooled/Pooled.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Pooled\Pooled.xlsx'
    type = type.lower()
    possibleTypes = ['car', 'suv', 'truck', 'all']
    if (type not in possibleTypes):
        print('Type not available.')
    else:
        sheetPrefix = 'weight-'
        if (type != 'all'):

            sheet = sheetPrefix + type
            data = p.read_excel(file, sheet)
            data.loc[data.loc[:, 'householdSize'] == '7 or more', 'householdSize'] = 7
            data = data.replace('I do not wish to answer', np.nan)
            data.loc[:, 'householdSize'] = data.loc[:, 'householdSize'].astype(float)
            data.loc[:, 'Vehicle Type'] = type
            data = appendCategorical(data)
            return data
        else:
            datas = []
            for tempType in possibleTypes[:len(possibleTypes) - 1]:
                tempData = getPooledData(tempType)
                datas.append(tempData)
            r = datas[0]

            for data in datas[1:]:
                r = r.append(data, sort=True)
            return r

def getMTurkData(type = 'all'):

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/MTurk/MTurkDataFull.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\MTurk\MTurkDataFull.xlsx'

    type = type.lower()
    possibleTypes = ['car', 'suv', 'all']
    if (type not in possibleTypes):
        print('Type not available.')
    else:
        sheetPrefix = 'weight-'
        if (type != 'all'):

            sheet = sheetPrefix + type
            data = p.read_excel(file, sheet)
            data.loc[data.loc[:, 'householdSize'] == '7 or more', 'householdSize'] = 7
            data = data.replace('I do not wish to answer', np.nan)
            data.loc[:, 'householdSize'] = data.loc[:, 'householdSize'].astype(float)
            data.loc[:, 'Vehicle Type'] = type
            data = appendCategorical(data)
            return data
        else:
            datas = []
            for tempType in possibleTypes[:len(possibleTypes) - 1]:
                tempData = getMTurkData(tempType)
                datas.append(tempData)
            r = datas[0]

            for data in datas[1:]:
                r = r.append(data, sort=True)
            return r

def getWeightedMTurkData(type = 'all'):

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/MTurk/MTurkDataWeighted.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\MTurk\MTurkDataWeighted.xlsx'

    type = type.lower()
    possibleTypes = ['car', 'suv', 'all']
    if (type not in possibleTypes):
        print('Type not available.')
    else:
        sheetPrefix = 'weight-'
        if (type != 'all'):

            sheet = sheetPrefix + type
            print(file)
            print(sheet)
            data = p.read_excel(file, sheet)
            data.loc[data.loc[:, 'householdSize'] == '7 or more', 'householdSize'] = 7
            data = data.replace('I do not wish to answer', np.nan)
            data.loc[:, 'householdSize'] = data.loc[:, 'householdSize'].astype(float)
            data.loc[:, 'Vehicle Type'] = type
            data = appendCategorical(data)
            return data
        else:
            datas = []
            for tempType in possibleTypes[:len(possibleTypes) - 1]:
                tempData = getWeightedMTurkData(tempType)
                datas.append(tempData)
            r = datas[0]

            for data in datas[1:]:
                r = r.append(data, sort=True)
            return r

def getDynataData(type = 'all'):

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Dynata/Dynata.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Dynata\Dynata.xlsx'

    type = type.lower()
    possibleTypes = ['car', 'suv', 'truck', 'all']
    if (type not in possibleTypes):
        print('Type not available.')
    else:
        sheetPrefix = 'weight-'
        if (type != 'all'):

            sheet = sheetPrefix + type
            data = p.read_excel(file, sheet)
            data.loc[data.loc[:, 'householdSize'] == '7 or more', 'householdSize'] = 7
            data = data.replace('I do not wish to answer', np.nan)
            data.loc[:, 'householdSize'] = data.loc[:, 'householdSize'].astype(float)
            data.loc[:, 'Vehicle Type'] = type
            data = appendCategorical(data)
            return data
        else:
            datas = []
            for tempType in possibleTypes[:len(possibleTypes) - 1]:
                tempData = getDynataData(tempType)
                datas.append(tempData)
            r = datas[0]

            for data in datas[1:]:
                r = r.append(data, sort=True)
            return r

def getWeightedDynataData(type = 'all'):

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Dynata/DynataWeighted.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Dynata\DynataWeighted.xlsx'

    type = type.lower()
    possibleTypes = ['car', 'suv', 'truck', 'all']
    if (type not in possibleTypes):
        print('Type not available.')
    else:
        sheetPrefix = 'weight-'
        if (type != 'all'):

            sheet = sheetPrefix + type
            data = p.read_excel(file, sheet)
            data.loc[data.loc[:, 'householdSize'] == '7 or more', 'householdSize'] = 7
            data = data.replace('I do not wish to answer', np.nan)
            data.loc[:, 'householdSize'] = data.loc[:, 'householdSize'].astype(float)
            data.loc[:, 'Vehicle Type'] = type
            data = appendCategorical(data)
            return data
        else:
            datas = []
            for tempType in possibleTypes[:len(possibleTypes) - 1]:
                tempData = getWeightedDynataData(tempType)
                datas.append(tempData)
            r = datas[0]

            for data in datas[1:]:
                r = r.append(data, sort=True)
            return r

def getHelvestonData(type = 'all'):

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Helveston/helvestonData.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Helveston\helvestonData.xlsx'

    type = type.lower()
    possibleTypes = ['car', 'suv', 'all']
    if (type not in possibleTypes):
        print('Type not available.')
    else:
        sheetPrefix = 'weight-'
        if (type != 'all'):

            sheet = sheetPrefix + type
            data = p.read_excel(file, sheet)
            data = data.replace('I do not wish to answer', np.nan)
            data.loc[data.loc[:, 'householdSize'] == '7 or more', 'householdSize'] = 7
            data = data.replace('I do not wish to answer', np.nan)
            data.loc[:, 'householdSize'] = data.loc[:, 'householdSize'].astype(float)
            data.loc[:, 'Vehicle Type'] = type

            fromYear = 2012
            toYear = 2020
            # infl = Inflator()
            # data.loc[:, 'Income'] = infl.inflateAll(data.loc[:, 'Income'], toYear=toYear, fromYear=fromYear)

            data = appendCategorical(data)
            return data
        else:
            datas = []
            for tempType in possibleTypes[:len(possibleTypes) - 1]:
                tempData = getHelvestonData(tempType)
                datas.append(tempData)
            r = datas[0]

            for data in datas[1:]:
                r = r.append(data, sort=True)
            return r

def getHelvestonMTurkData(type = 'all'):

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Helveston/helvestonDataMTurk.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Helveston\helvestonDataMTurk.xlsx'

    type = type.lower()
    possibleTypes = ['car', 'suv', 'all']
    if (type not in possibleTypes):
        print('Type not available.')
    else:
        sheetPrefix = 'weight-'
        if (type != 'all'):

            sheet = sheetPrefix + type
            data = p.read_excel(file, sheet)
            data = data.replace('I do not wish to answer', np.nan)
            data.loc[data.loc[:, 'householdSize'] == '7 or more', 'householdSize'] = 7
            data = data.replace('I do not wish to answer', np.nan)
            data.loc[:, 'householdSize'] = data.loc[:, 'householdSize'].astype(float)
            data.loc[:, 'Vehicle Type'] = type

            fromYear = 2012
            toYear = 2020
            # infl = Inflator()
            # data.loc[:, 'Income'] = infl.inflateAll(data.loc[:, 'Income'], toYear=toYear, fromYear=fromYear)

            data = appendCategorical(data)
            return data
        else:
            datas = []
            for tempType in possibleTypes[:len(possibleTypes) - 1]:
                tempData = getHelvestonMTurkData(tempType)
                datas.append(tempData)
            r = datas[0]

            for data in datas[1:]:
                r = r.append(data, sort=True)
            return r

def getWeightedPooledData(type = 'all'):

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Pooled/PooledWeighted.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Pooled\PooledWeighted.xlsx'
    type = type.lower()
    possibleTypes = ['car', 'suv', 'truck', 'all']
    if (type not in possibleTypes):
        print('Type not available.')
    else:
        sheetPrefix = 'weight-'
        if (type != 'all'):

            sheet = sheetPrefix + type
            data = p.read_excel(file, sheet)
            data.loc[data.loc[:, 'householdSize'] == '7 or more', 'householdSize'] = 7
            data = data.replace('I do not wish to answer', np.nan)
            data.loc[:, 'householdSize'] = data.loc[:, 'householdSize'].astype(float)
            data.loc[:, 'Vehicle Type'] = type
            data = appendCategorical(data)
            return data
        else:
            datas = []
            for tempType in possibleTypes[:len(possibleTypes) - 1]:
                tempData = getWeightedPooledData(tempType)
                datas.append(tempData)
            r = datas[0]

            for data in datas[1:]:
                r = r.append(data, sort=True)
            return r

def getWeightedHelvestonData(type = 'all'):

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Helveston/helvestonDataWeighted.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Helveston\helvestonDataWeighted.xlsx'

    type = type.lower()
    possibleTypes = ['car', 'suv', 'all']
    if (type not in possibleTypes):
        print('Type not available.')
    else:
        sheetPrefix = 'weight-'
        if (type != 'all'):

            sheet = sheetPrefix + type
            data = p.read_excel(file, sheet)
            data = data.replace('I do not wish to answer', np.nan)
            data.loc[data.loc[:, 'householdSize'] == '7 or more', 'householdSize'] = 7
            data = data.replace('I do not wish to answer', np.nan)
            data.loc[:, 'householdSize'] = data.loc[:, 'householdSize'].astype(float)
            data.loc[:, 'Vehicle Type'] = type

            fromYear = 2012
            toYear = 2020
            # infl = Inflator()
            # data.loc[:, 'Income'] = infl.inflateAll(data.loc[:, 'Income'], toYear=toYear, fromYear=fromYear)

            data = appendCategorical(data)
            return data
        else:
            datas = []
            for tempType in possibleTypes[:len(possibleTypes) - 1]:
                tempData = getWeightedHelvestonData(tempType)
                datas.append(tempData)
            r = datas[0]

            for data in datas[1:]:
                r = r.append(data, sort=True)
            return r

def loadPooledWorkbook():

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Pooled/Pooled.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Pooled\Pooled.xlsx'

    r = p.read_excel(file, None)

    return r

def loadMTurkWorkbook():

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/MTurk/MTurkDataFull.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\MTurk\MTurkDataFull.xlsx'
    r = p.read_excel(file, None)

    return r

def loadDynataWorkbook():

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Dynata/Dynata.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Dynata\Dynata.xlsx'

    r = p.read_excel(file, None)

    return r

def loadHelvestonWorkbook():

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Helveston/helvestonData.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Helveston\helvestonData.xlsx'

    r = p.read_excel(file, None)

    return r

def loadHelvestonMTurkWorkbook():

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Helveston/helvestonDataMTurk.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Helveston\helvestonDataMTurk.xlsx'

    r = p.read_excel(file, None)

    return r

def appendCategorical(data):
    #Woman
    data.loc[:, 'Woman'] = 0
    data.loc[data.loc[:, 'gender']=='Woman', 'Woman'] = 1
    #College grad
    data.loc[:, 'CollegeGrad'] = 0
    collegeGradOptions = ['4 year university degree', 'Masters degree', 'Doctoral degree', '4 year university degree (bachelors)']
    data.loc[data.loc[:, 'education'].isin(collegeGradOptions), 'CollegeGrad'] = 1
    #Single
    data.loc[:, 'Single'] = 0
    singleOptions = ['Single', 'Widowed', 'Divorced', 'Separated']
    data.loc[data.loc[:, 'livingSit'].isin(singleOptions), 'Single'] = 1
    #White
    # data.loc[:, 'White'] = 0
    # data.loc[data.loc[:, 'race']=='Caucasian', 'White'] = 1
    #Decade
    data.loc[:, 'Decade'] = np.round(data.loc[:, 'Age'], -1)
    #

    return data

def appendCategoricalMaritz(data, year2012=True):
    data.loc[:, 'Decade'] = np.round(data.loc[:, 'Age'], -1)

    if('Weight' not in data.columns):
        data.loc[:, 'Weight'] = data.loc[:, 'weight']
    if(year2012):
        # Woman
        data.loc[:, 'Woman'] = 0
        data.loc[data.loc[:, 'gender'] == 'Woman', 'Woman'] = 1
        # College grad
        data.loc[:, 'CollegeGrad'] = 0
        collegeGradOptions = ['4year', '>4year']
        data.loc[data.loc[:, 'education'].isin(collegeGradOptions), 'CollegeGrad'] = 1
        # Single
        data.loc[:, 'Single'] = 0
        singleOptions = ['single', 'divorced']
        data.loc[data.loc[:, 'livingSit'].isin(singleOptions), 'Single'] = 1
    else:
        # Woman
        data.loc[:, 'Woman'] = 0
        data.loc[data.loc[:, 'cleanGender'] == 'Female', 'Woman'] = 1
        data.loc[data.loc[:, 'gender'].isna(), 'Woman'] = np.nan
        # College grad
        data.loc[:, 'CollegeGrad'] = 0
        collegeGradOptions = ["Graduate with bachelor's degree", 'Postgraduate degree', 'Some postgraduate study']
        data.loc[data.loc[:, 'cleanEducation'].isin(collegeGradOptions), 'CollegeGrad'] = 1
        data.loc[data.loc[:, 'education'].isna(), 'CollegeGrad'] = np.nan
        # Single
        data.loc[:, 'Single'] = 0
        singleOptions = ['single', 'divorced']
        data.loc[data.loc[:, 'cleanMarital'].isin(singleOptions), 'Single'] = 1
        data.loc[data.loc[:, 'maritalStatus'].isna(), 'Single'] = np.nan
    return data

def getSummaryData(data, vars, weightCom = None):
    stata.run('clear all')

    stata.pdataframe_to_data(data)
    varString = ''
    for var in vars:
        varString+='{} '.format(var)
    if(weightCom==None):
        stata.run('qui tabstat {}, statistics(mean count sd) save'.format(varString))
    else:
        stata.run('qui tabstat {} [{}], statistics(mean count sd) save'.format(varString, weightCom))
    r = stata.get_return()
    r = r[list(r.keys())[0]]
    r = p.DataFrame(r, columns=vars, index=['mean', 'count', 'sd'])
    return r

def cleanVarName(varName):
    badNames = ['householdSize', 'Age', 'Income', 'Woman', 'CollegeGrad', 'Single']
    cleanNames = ['Household Size', 'Age', 'Income', 'Woman', 'College Grad', 'Single']
    mapping = dict(zip(badNames, cleanNames))

    if(varName in mapping.keys()):
        return mapping[varName]
    else:
        return varName

def cleanSingleResult(data, name):
    newCol = []
    nameCol = []
    for col in data.columns:
        if(data.loc['mean', col]<10):
            numDigits = 2
        else:
            numDigits = 1
        newCol.append(np.round(data.loc['mean', col],numDigits))
        newCol.append('('+str(np.round(data.loc['sd', col],numDigits))+')')
        nameCol.append(cleanVarName(col))
        nameCol.append('')

    nameCol = np.array(nameCol).reshape((-1,1))
    newCol = np.array(newCol).reshape((-1, 1))
    dfData = np.hstack((nameCol, newCol))
    r = p.DataFrame(dfData, columns=['Variable', name])
    return r

def buildFullTable(datas, names=None):
    if(type(datas)!=dict and type(datas)!=type(OrderedDict())):
        if(names==None):
            print("You must specify names for the table")
            return None
        else:
            datas = OrderedDict(zip(datas, names))

    formattedCols = []

    for tempName, tempData in datas.items():
        tempFormattedCol = cleanSingleResult(tempData, tempName)
        formattedCols.append(tempFormattedCol)

    r = formattedCols[0]

    for formattedCol in formattedCols[1:]:
        tempCols = formattedCol.columns
        relCol = tempCols[-1]
        r.loc[:, relCol] = formattedCol.loc[:, relCol]

    return r

def loadMaritz2010(vehType=None):

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/Weighting/CleanDistributions/Maritz2010.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\Weighting\CleanDistributions\Maritz2010.xlsx'

    data = p.read_excel(file)
    data.loc[:, 'vehicleTypeClean'] = data.loc[:, 'vehicleTypeClean'].str.lower()
    if(vehType!=None):
        data = data.loc[data.loc[:, 'vehicleTypeClean']==vehType, :]

    data = appendCategoricalMaritz(data)
    infl = Inflator()
    fromYear = 2010
    toYear = 2020
    data.loc[:, 'Income'] = infl.inflateAll(data.loc[:, 'Income'], fromYear, toYear)

    return data

def loadMaritz2012():
    file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/Weighting/CleanDistributions/Maritz2012.xlsx'
    data = p.read_excel(file)
    data = appendCategoricalMaritz(data)
    infl = Inflator()
    fromYear = 2012
    toYear = 2020
    data.loc[:, 'Income'] = infl.inflateAll(data.loc[:, 'Income'], fromYear, toYear)

    return data

def loadMaritz2018(vehType=None):

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/Data/EV Preferences Data/ForsytheMaritz2018/Maritz2018JointDistribution.csv'
    else:
        file = r'C:\Users\Connor\Box\CMU\Data\EV Preferences Data\ForsytheMaritz2018\Maritz2018JointDistribution.csv'

    data = p.read_csv(file)
    data.loc[:, 'vehicleTypeClean'] = data.loc[:, 'vehicleTypeClean'].str.lower()
    if (vehType != None):
        data = data.loc[data.loc[:, 'vehicleTypeClean'] == vehType, :]

    renameDict = {'cleanIncome':'Income', 'age':'Age', 'numHousehold':'householdSize'}
    data = data.rename(renameDict, axis=1)
    data = appendCategoricalMaritz(data, False)
    infl = Inflator()
    fromYear = 2018
    toYear = 2020
    data.loc[:, 'Income'] = infl.inflateAll(data.loc[:, 'Income'], fromYear, toYear)
    return data

def getWeightingDistribution(baseData, variables):
    if(type(variables)==str):
        variables = [variables]

    baseData.loc[:, 'Count'] = 1

    r = baseData.groupby(variables, as_index=False).sum()

    r.loc[:, 'pdf'] = r.loc[:, 'Weight']/np.sum(r.loc[:, 'Weight'])

    keepCols = deepcopy(variables)

    keepCols.extend(['Weight', 'pdf', 'Count'])

    return r.loc[:, keepCols]

def joinMaritzSampleDistributions(maritzDist, sampleDist, weightingVars):
    keepVars = deepcopy(weightingVars)
    keepVars.append('pdf')

    # maritzDist = maritzDist.loc[: keepVars]
    # sampleDist = sampleDist.loc[: keepVars]

    jointDist = maritzDist.merge(sampleDist, on=weightingVars, suffixes=('Maritz', 'Sample'), how='right', indicator=True)

    # print(list(jointDist.loc[:, 'pdfMaritz']))

    # print(type(jointDist))

    jointDist.loc[jointDist.loc[:, 'pdfMaritz'] != jointDist.loc[:, 'pdfMaritz'], 'pdfMaritz'] = min(
    jointDist.loc[:, 'pdfMaritz']) / 10000



    return jointDist

def getKLDivergenceFromDists(dists):
    emp = np.array(dists['pdfSample']).reshape((-1,1))
    des = np.array(dists['pdfMaritz']).reshape((-1,1))

    div = emp/des
    logDiv = np.log(div)

    kl = np.transpose(emp)@logDiv

    return float(kl)

def getDistributions(personalData, desiredData, weightingCols, weights):
    if(len(weights)!=len(personalData)):
        print('Weights do not equal the number of respondents.')
    personalData.loc[:, 'Weight'] = weights

    maritzDist = getWeightingDistribution(desiredData, weightingCols)
    sampleDist = getWeightingDistribution(personalData, weightingCols)

    jointDist = joinMaritzSampleDistributions(maritzDist, sampleDist, weightingCols)

    return jointDist


def getKLDivergence(personalData, desiredData, cols, weights):

    dists = getDistributions(personalData, desiredData, cols, weights)
    kl = getKLDivergenceFromDists(dists)
    return kl

def getWeights(personalData, desiredData, cols, kappa=5, maxiter=1000, uniformX0=True, total=1, ftol=1e-10):

    nPoints = len(personalData)
    # total = nPoints
    if(kappa==None or kappa==0):
        lb = -np.inf
        ub = np.inf
    else:
        lb = total / (kappa * nPoints)
        ub = (total / nPoints)*kappa
    numZeros = str(lb).count('0')

    bounds = Bounds(lb=[lb] * nPoints, ub=[ub] * nPoints)
    boundsLinear = LinearConstraint(np.diag(np.array([1] * nPoints)), lb=lb, ub=ub)
    sumCons = LinearConstraint(np.array([1] * nPoints).reshape((1, -1)), lb=total, ub=total)
    if(uniformX0):
        x0 = [total / nPoints] * nPoints
    else:
        x0 = np.random.uniform(lb, ub, nPoints)

    fun = lambda x: getKLDivergence(personalData, desiredData, cols, x)
    print("Running optimization. Number of Weights {}. Sum of Weights={}. Weights bound on {}".format(nPoints, total, [lb, ub]))
    method = 'SLSQP'
    if(method=='SLSQP'):
        res = minimize(fun, x0, constraints=[sumCons, boundsLinear], method=method, options={'disp': False, 'maxiter':maxiter, 'ftol':ftol})
    else:
        res = minimize(fun, x0, constraints=[sumCons, boundsLinear], method=method,
                       options={'disp': False, 'maxiter': maxiter, 'gtol': ftol})
    weightSum = np.sum(res.x)
    minWeight = np.min(res.x)
    maxWeight = np.max(res.x)
    print('Optimization complete. Final Weights Sum = {}. Min, Max Weight = {}'.format(weightSum, [minWeight, maxWeight]))
    newDists = getDistributions(personalData, desiredData, cols, res.x)


    r = {'weights':res.x*len(res.x), 'result':res, 'dists':newDists}

    return r

def assignIncomeGroups(maritzData, sampleData):
    maritzIncomeLevels = sorted(list(set(maritzData.loc[maritzData.loc[:, 'Income']==maritzData.loc[:, 'Income'], 'Income'])))
    sampleIncomeLevels = sorted(list(set(sampleData.loc[sampleData.loc[:, 'Income']==sampleData.loc[:, 'Income'], 'Income'])))

    maritzData.loc[:, 'IncomeGroup'] = np.nan
    sampleData.loc[:, 'IncomeGroup'] = np.nan

    for i in range(len(maritzIncomeLevels)):
        tempIncomeLevel = maritzIncomeLevels[i]
        maritzData.loc[maritzData.loc[:, 'Income']==tempIncomeLevel, 'IncomeGroup'] = i

    newMapping = OrderedDict()

    for incomeLevel in sampleIncomeLevels:
        sumSqVal = np.inf
        groupVal = -1
        for i in range(len(maritzIncomeLevels)):
            tempIncomeLevel = maritzIncomeLevels[i]
            tempSumSq = (incomeLevel-tempIncomeLevel)**2
            if(tempSumSq<sumSqVal):
                groupVal = i
                sumSqVal = tempSumSq
        newMapping[incomeLevel] = groupVal



    for tempIncome, groupNumber in newMapping.items():
        sampleData.loc[sampleData.loc[:, 'Income'] == tempIncome, 'IncomeGroup'] = groupNumber

    maxSampleIncomeGroup = max(newMapping.values())

    maritzData.loc[maritzData.loc[:, 'IncomeGroup']>maxSampleIncomeGroup, 'IncomeGroup'] = maxSampleIncomeGroup



    return maritzData, sampleData
def weightPooledData(weightingVars, maritz2018=True, weightName='Weight', save=True):

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Pooled/PooledWeighted.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Pooled\PooledWeighted.xlsx'

    types = ['car', 'suv', 'truck']
    # types = ['truck']
    weightingOutcomes = {}
    print('Loading Maritz Data')
    if(maritz2018):
        maritzData = loadMaritz2018()
    else:
        maritzData = loadMaritz2010()

    incomeWeighted = 'Income' in weightingVars
    oldWeightingVars = deepcopy(weightingVars)
    if(incomeWeighted):
        weightingVars = deepcopy(weightingVars)
        weightingVars.remove('Income')
        weightingVars.append('IncomeGroup')

    for type in types:
        print('Loading {}s data'.format(type))
        tempPersonalData = getPooledData(type)
        maritzData = loadMaritzData(maritz2018, type)
        print('{} data loaded'.format(type))
        print('Working on pooled data weights for {}s.'.format(type))
        if(incomeWeighted):
            maritzData, tempPersonalData = assignIncomeGroups(maritzData, tempPersonalData)

        maritzData = dropNANDemogEntries(maritzData, weightingVars)
        tempPersonalData = dropNANDemogEntries(tempPersonalData, weightingVars)
        tempOutcome = getWeights(tempPersonalData, maritzData, weightingVars)
        tempMappingDF = deepcopy(tempPersonalData)
        tempMappingDF.loc[:, 'Weight'] = tempOutcome['weights']
        tempOutcome['mappingDF'] = tempMappingDF.loc[:, ['ID', 'Weight']]
        tempOutcome['mappingDF'] = tempOutcome['mappingDF'].rename({'Weight':weightName},axis=1)
        print('Completed weighting pooled {}s data.'.format(type))
        weightingOutcomes[type] = tempOutcome
    if(save):
        print('Loading full pooled data workbook.')
        fullWorkbook = loadPooledWorkbook()
        newData = {}
        print('Appending weights across workbook')

        for type in types:
            weightSheet = 'weight-{}'.format(type)
            cbcShortSheet = 'cbcShort-{}'.format(type)
            cbcLongSheet = 'cbcLong-{}'.format(type)
            demoSheet = 'demo-{}'.format(type)

            idWeightDF = weightingOutcomes[type]['mappingDF']

            newData[weightSheet] = deepcopy(fullWorkbook[weightSheet])
            newData[weightSheet] = newData[weightSheet].merge(idWeightDF, on='ID', how='right')


            newData[cbcShortSheet] = deepcopy(fullWorkbook[cbcShortSheet])
            newData[cbcShortSheet] = newData[cbcShortSheet].merge(idWeightDF, on='ID')

            newData[cbcLongSheet] = deepcopy(fullWorkbook[cbcLongSheet])
            newData[cbcLongSheet] = newData[cbcLongSheet].merge(idWeightDF, on='ID')

            newData[demoSheet] = deepcopy(fullWorkbook[demoSheet])
            newData[demoSheet] = newData[demoSheet].rename({'id':'ID'}, axis=1)
            newData[demoSheet] = newData[demoSheet].merge(idWeightDF, on='ID')

        print('Saving new workbook')
        excelWriter = p.ExcelWriter(file)
        for sheet, data in newData.items():
            data.to_excel(excelWriter, sheet, index=False)

        excelWriter.close()
        print('Workbook saved')

    return weightingOutcomes

def weightMTurkData(weightingVars, maritz2018=True, weightName='Weight', save=True, weightNames = None):


    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/MTurk/MTurkDataWeighted.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\MTurk\MTurkDataWeighted.xlsx'
    types = ['car', 'suv']
    # types = ['suv']
    weightingOutcomes = {}
    print('Loading Maritz Data')
    if(maritz2018):
        maritzData = loadMaritz2018()
    else:
        maritzData = loadMaritz2010()


    incomeWeighted = 'Income' in weightingVars
    oldWeightingVars = deepcopy(weightingVars)
    if(incomeWeighted):
        weightingVars = deepcopy(weightingVars)
        weightingVars.remove('Income')
        weightingVars.append('IncomeGroup')

    for type in types:
        print('Loading {}s data'.format(type))
        tempPersonalData = getMTurkData(type)
        maritzData = loadMaritzData(maritz2018, type)
        print('{} data loaded'.format(type))
        print('Working on MTurk data weights for {}s.'.format(type))
        if(incomeWeighted):
            maritzData, tempPersonalData = assignIncomeGroups(maritzData, tempPersonalData)

        maritzData = dropNANDemogEntries(maritzData, weightingVars)
        tempPersonalData = dropNANDemogEntries(tempPersonalData, weightingVars)
        tempOutcome = getWeights(tempPersonalData, maritzData, weightingVars)
        tempMappingDF = deepcopy(tempPersonalData)
        tempMappingDF.loc[:, 'Weight'] = tempOutcome['weights']
        tempOutcome['mappingDF'] = tempMappingDF.loc[:, ['ID', 'Weight']]
        tempOutcome['mappingDF'] = tempOutcome['mappingDF'].rename({'Weight':weightName},axis=1)
        print('Completed weighting MTurk {}s data.'.format(type))
        weightingOutcomes[type] = tempOutcome
    if(save):
        print('Loading full MTurk data workbook.')
        fullWorkbook = loadMTurkWorkbook()
        newData = {}
        print('Appending weights across workbook')

        for type in types:
            weightSheet = 'weight-{}'.format(type)
            cbcShortSheet = 'cbcShort-{}'.format(type)
            cbcLongSheet = 'cbcLong-{}'.format(type)
            demoSheet = 'demo-{}'.format(type)

            idWeightDF = weightingOutcomes[type]['mappingDF']

            newData[weightSheet] = deepcopy(fullWorkbook[weightSheet])
            newData[weightSheet] = newData[weightSheet].merge(idWeightDF, on='ID', how='right')


            newData[cbcShortSheet] = deepcopy(fullWorkbook[cbcShortSheet])
            newData[cbcShortSheet] = newData[cbcShortSheet].merge(idWeightDF, on='ID')

            newData[cbcLongSheet] = deepcopy(fullWorkbook[cbcLongSheet])
            newData[cbcLongSheet] = newData[cbcLongSheet].merge(idWeightDF, on='ID')

            newData[demoSheet] = deepcopy(fullWorkbook[demoSheet])
            newData[demoSheet] = newData[demoSheet].rename({'id':'ID'}, axis=1)
            newData[demoSheet] = newData[demoSheet].merge(idWeightDF, on='ID')

        print('Saving new workbook')
        excelWriter = p.ExcelWriter(file)
        for sheet, data in newData.items():
            data.to_excel(excelWriter, sheet, index=False)

        excelWriter.close()
        print('Workbook saved')

    return weightingOutcomes

def weightDynataData(weightingVars, maritz2018=True, weightName='Weight', save=True):

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Dynata/DynataWeighted.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Dynata\DynataWeighted.xlsx.xlsx'

    types = ['car', 'suv', 'truck']
    # types = ['truck']
    weightingOutcomes = {}
    print('Loading Maritz Data')
    if(maritz2018):
        maritzData = loadMaritz2018()
    else:
        maritzData = loadMaritz2010()

    incomeWeighted = 'Income' in weightingVars
    oldWeightingVars = deepcopy(weightingVars)
    if(incomeWeighted):
        weightingVars = deepcopy(weightingVars)
        weightingVars.remove('Income')
        weightingVars.append('IncomeGroup')

    for type in types:
        print('Loading {}s data'.format(type))
        tempPersonalData = getDynataData(type)
        maritzData = loadMaritzData(maritz2018, type)
        print('{} data loaded'.format(type))
        print('Working on Dynata data weights for {}s.'.format(type))
        if(incomeWeighted):
            maritzData, tempPersonalData = assignIncomeGroups(maritzData, tempPersonalData)

        maritzData = dropNANDemogEntries(maritzData, weightingVars)
        tempPersonalData = dropNANDemogEntries(tempPersonalData, weightingVars)
        tempOutcome = getWeights(tempPersonalData, maritzData, weightingVars)
        tempMappingDF = deepcopy(tempPersonalData)
        tempMappingDF.loc[:, 'Weight'] = tempOutcome['weights']
        tempOutcome['mappingDF'] = tempMappingDF.loc[:, ['ID', 'Weight']]
        tempOutcome['mappingDF'] = tempOutcome['mappingDF'].rename({'Weight':weightName},axis=1)
        print('Completed weighting Dynata {}s data.'.format(type))
        weightingOutcomes[type] = tempOutcome
    if(save):
        print('Loading full Dynata data workbook.')
        fullWorkbook = loadDynataWorkbook()
        newData = {}
        print('Appending weights across workbook')

        for type in types:
            weightSheet = 'weight-{}'.format(type)
            cbcShortSheet = 'cbcShort-{}'.format(type)
            cbcLongSheet = 'cbcLong-{}'.format(type)
            demoSheet = 'demo-{}'.format(type)

            idWeightDF = weightingOutcomes[type]['mappingDF']

            newData[weightSheet] = deepcopy(fullWorkbook[weightSheet])
            newData[weightSheet] = newData[weightSheet].merge(idWeightDF, on='ID', how='right')


            newData[cbcShortSheet] = deepcopy(fullWorkbook[cbcShortSheet])
            newData[cbcShortSheet] = newData[cbcShortSheet].merge(idWeightDF, on='ID')

            newData[cbcLongSheet] = deepcopy(fullWorkbook[cbcLongSheet])
            newData[cbcLongSheet] = newData[cbcLongSheet].merge(idWeightDF, on='ID')

            newData[demoSheet] = deepcopy(fullWorkbook[demoSheet])
            newData[demoSheet] = newData[demoSheet].rename({'id':'ID'}, axis=1)
            newData[demoSheet] = newData[demoSheet].merge(idWeightDF, on='ID')

        print('Saving new workbook')
        excelWriter = p.ExcelWriter(file)
        for sheet, data in newData.items():
            data.to_excel(excelWriter, sheet, index=False)

        excelWriter.close()
        print('Workbook saved')

    return weightingOutcomes



def weightHelvestonData(weightingVars, maritz2018=False, weightName='Weight', save=True):

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Helveston/helvestonDataWeighted.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Helveston\helvestonDataWeighted.xlsx'

    types = ['car', 'suv']
    weightingOutcomes = {}
    print('Loading Maritz Data')
    if(maritz2018):
        maritzData = loadMaritz2018()
    else:
        maritzData = loadMaritz2010()

    incomeWeighted = 'Income' in weightingVars
    oldWeightingVars = deepcopy(weightingVars)
    if(incomeWeighted):
        weightingVars = deepcopy(weightingVars)
        weightingVars.remove('Income')
        weightingVars.append('IncomeGroup')

    for type in types:
        print('Loading {}s data'.format(type))
        tempPersonalData = getHelvestonData(type)
        maritzData = loadMaritzData(maritz2018, type)
        print('{} data loaded'.format(type))
        print('Working on helveston pooled data weights for {}s.'.format(type))
        if(incomeWeighted):
            maritzData, tempPersonalData = assignIncomeGroups(maritzData, tempPersonalData)

        maritzData = dropNANDemogEntries(maritzData, weightingVars)
        tempPersonalData = dropNANDemogEntries(tempPersonalData, weightingVars)
        tempOutcome = getWeights(tempPersonalData, maritzData, weightingVars)
        tempMappingDF = deepcopy(tempPersonalData)
        tempMappingDF.loc[:, 'Weight'] = tempOutcome['weights']
        tempOutcome['mappingDF'] = tempMappingDF.loc[:, ['ID', 'Weight']]
        tempOutcome['mappingDF'] = tempOutcome['mappingDF'].rename({'Weight':weightName},axis=1)
        print('Completed weighting helveston pooled {}s data.'.format(type))
        weightingOutcomes[type] = tempOutcome
    if(save):
        print('Loading full helveston pooled data workbook.')
        fullWorkbook = loadHelvestonWorkbook()
        newData = {}
        print('Appending weights across workbook')

        for type in types:
            weightSheet = 'weight-{}'.format(type)
            cbcShortSheet = 'cbcShort-{}'.format(type)
            cbcLongSheet = 'cbcLong-{}'.format(type)
            demoSheet = 'demo-{}'.format(type)

            idWeightDF = weightingOutcomes[type]['mappingDF']

            newData[weightSheet] = deepcopy(fullWorkbook[weightSheet])
            newData[weightSheet] = newData[weightSheet].merge(idWeightDF, on='ID', how='right')


            newData[cbcShortSheet] = deepcopy(fullWorkbook[cbcShortSheet])
            newData[cbcShortSheet] = newData[cbcShortSheet].merge(idWeightDF, on='ID')

            newData[cbcLongSheet] = deepcopy(fullWorkbook[cbcLongSheet])
            newData[cbcLongSheet] = newData[cbcLongSheet].merge(idWeightDF, on='ID')

            newData[demoSheet] = deepcopy(fullWorkbook[demoSheet])
            newData[demoSheet] = newData[demoSheet].rename({'id':'ID'}, axis=1)
            newData[demoSheet] = newData[demoSheet].merge(idWeightDF, on='ID')

        print('Saving new workbook')
        excelWriter = p.ExcelWriter(file)
        for sheet, data in newData.items():
            data.to_excel(excelWriter, sheet, index=False)

        excelWriter.close()
        print('Workbook saved')

    return weightingOutcomes

def weightHelvestonMTurkData(weightingVars, maritz2018=False, weightName='Weight', save=True):

    if (sys.platform == 'darwin'):
        file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Helveston/helvestonDataMTurkWeighted.xlsx'
    else:
        file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Helveston\helvestonDataMTurkWeighted.xlsx'

    types = ['car', 'suv']
    weightingOutcomes = {}
    print('Loading Maritz Data')
    if(maritz2018):
        maritzData = loadMaritz2018()
    else:
        maritzData = loadMaritz2010()

    incomeWeighted = 'Income' in weightingVars
    oldWeightingVars = deepcopy(weightingVars)
    if(incomeWeighted):
        weightingVars = deepcopy(weightingVars)
        weightingVars.remove('Income')
        weightingVars.append('IncomeGroup')

    for type in types:
        print('Loading {}s data'.format(type))
        tempPersonalData = getHelvestonMTurkData(type)
        maritzData = loadMaritzData(maritz2018, type)
        print('{} data loaded'.format(type))
        print('Working on helveston pooled data weights for {}s.'.format(type))
        if(incomeWeighted):
            maritzData, tempPersonalData = assignIncomeGroups(maritzData, tempPersonalData)

        maritzData = dropNANDemogEntries(maritzData, weightingVars)
        tempPersonalData = dropNANDemogEntries(tempPersonalData, weightingVars)
        tempOutcome = getWeights(tempPersonalData, maritzData, weightingVars)
        tempMappingDF = deepcopy(tempPersonalData)
        tempMappingDF.loc[:, 'Weight'] = tempOutcome['weights']
        tempOutcome['mappingDF'] = tempMappingDF.loc[:, ['ID', 'Weight']]
        tempOutcome['mappingDF'] = tempOutcome['mappingDF'].rename({'Weight':weightName},axis=1)
        print('Completed weighting helveston Mturk {}s data.'.format(type))
        weightingOutcomes[type] = tempOutcome
    if(save):
        print('Loading full helveston MTurk data workbook.')
        fullWorkbook = loadHelvestonMTurkWorkbook()
        newData = {}
        print('Appending weights across workbook')

        for type in types:
            weightSheet = 'weight-{}'.format(type)
            cbcShortSheet = 'cbcShort-{}'.format(type)
            cbcLongSheet = 'cbcLong-{}'.format(type)
            demoSheet = 'demo-{}'.format(type)

            idWeightDF = weightingOutcomes[type]['mappingDF']

            newData[weightSheet] = deepcopy(fullWorkbook[weightSheet])
            newData[weightSheet] = newData[weightSheet].merge(idWeightDF, on='ID', how='right')


            newData[cbcShortSheet] = deepcopy(fullWorkbook[cbcShortSheet])
            newData[cbcShortSheet] = newData[cbcShortSheet].merge(idWeightDF, on='ID')

            newData[cbcLongSheet] = deepcopy(fullWorkbook[cbcLongSheet])
            newData[cbcLongSheet] = newData[cbcLongSheet].merge(idWeightDF, on='ID')

            newData[demoSheet] = deepcopy(fullWorkbook[demoSheet])
            newData[demoSheet] = newData[demoSheet].rename({'id':'ID'}, axis=1)
            newData[demoSheet] = newData[demoSheet].merge(idWeightDF, on='ID')

        print('Saving new workbook')
        excelWriter = p.ExcelWriter(file)
        for sheet, data in newData.items():
            data.to_excel(excelWriter, sheet, index=False)

        excelWriter.close()
        print('Workbook saved')

    return weightingOutcomes

def dropNANDemogEntries(data, weightingVars):
    for weightingVar in weightingVars:
        data = data.loc[data.loc[:, weightingVar]==data.loc[:, weightingVar],:]
        if(weightingVar in ['Age', 'Decade']):
            data = data.loc[data.loc[:, 'Age']<130, :]
            data = data.loc[data.loc[:, 'Age']==data.loc[:, 'Age'],:]

    return data

def buildSummaryTableFromData(datas, baseData, sumVars, weightVars, fileName, weightCom='w=Weight2018'):
    sumDatas = OrderedDict()
    for dataType, tempData in datas.items():
        tempUnweighted = getSummaryData(tempData, sumVars)
        tempWeighted = getSummaryData(tempData, sumVars, weightCom)
        if(len(datas)==1):
            sumDatas['\makecell{Unweighted}'] = tempUnweighted
            sumDatas['\makecell{Weighted}'] = tempWeighted
        else:
            sumDatas['\makecell{' + '{}'.format(dataType.capitalize()) + '\\\\ Unweighted}'] = tempUnweighted
            sumDatas['\makecell{' + '{}'.format(dataType.capitalize()) + '\\\\ Weighted}'] = tempWeighted

    for name, data in baseData.items():
        newData = dropNANDemogEntries(data, weightVars)
        tempWeighted = getSummaryData(newData, sumVars, 'w=Weight')
        sumDatas[name] = tempWeighted

    tableDF = buildFullTable(sumDatas)
    tableDF.to_latex(fileName, escape=False, index=False)
    return tableDF



def constructFullWeightMappingDF(weightingResults):
    typesBuilt = []
    r = OrderedDict()
    for id, typeDict in weightingResults.items():
        for type, rDict in typeDict.items():
            if(type not in typesBuilt):
                r[type] = rDict['mappingDF']
                typesBuilt.append(type)
            else:
                r[type] = r[type].merge(rDict['mappingDF'], on='ID', how='outer')

    return r

def weightMultiple(weightingVarLists, maritz2018List, names, pooled=True, mturk=False, dynata=False, helveston=False, helvestonMTurk=False, save=True):

    pooledResults = OrderedDict()
    dynataResults = OrderedDict()
    mturkResults = OrderedDict()
    helvestonResults = OrderedDict()
    helvestonMTurkResults = OrderedDict()

    if (pooled):
        print('Weighting Pooled repsondents')
        argList = []
        for weightingVars, maritz2018, name in zip(weightingVarLists, maritz2018List, names):

            print('Working on weighting the following demographics: {}'.format(weightingVars))
            argList.append([weightingVars, maritz2018, name, False])
            tempRes = weightPooledData(weightingVars, maritz2018, name, save=False)
            pooledResults[(name, tuple(weightingVars), maritz2018)] = tempRes

        print('Completed weighting Pooled respondents')

        if (save):
            pooledDFDict = constructFullWeightMappingDF(pooledResults)
            saveNewWeights(pooledDFDict, pooled=True)

    if(mturk):
        print('Weighting MTurk repsondents')
        argList = []
        for weightingVars, maritz2018, name in zip(weightingVarLists, maritz2018List, names):

            print('Working on weighting the following demographics: {}'.format(weightingVars))
            argList.append([weightingVars, maritz2018, name, False])
            tempRes = weightMTurkData(weightingVars, maritz2018, name, save=False)
            mturkResults[(name, tuple(weightingVars), maritz2018)] = tempRes

        print('Completed weighting MTurk respondents')

        if(save):
            mturkDFDict = constructFullWeightMappingDF(mturkResults)
            saveNewWeights(mturkDFDict, mturk=True)

    if (dynata):
        print('Weighting Dynata repsondents')
        argList = []
        for weightingVars, maritz2018, name in zip(weightingVarLists, maritz2018List, names):

            print('Working on weighting the following demographics: {}'.format(weightingVars))
            argList.append([weightingVars, maritz2018, name, False])
            tempRes = weightDynataData(weightingVars, maritz2018, name, save=False)
            dynataResults[(name, tuple(weightingVars), maritz2018)] = tempRes

        print('Completed weighting Dynata respondents')

        if (save):
            dynataDFDict = constructFullWeightMappingDF(dynataResults)
            saveNewWeights(dynataDFDict, dynata=True)

    if(helveston):
        print('Weighting Helveston repsondents')
        argList = []
        for weightingVars, maritz2018, name in zip(weightingVarLists, maritz2018List, names):

            print('Working on weighting the following demographics: {}'.format(weightingVars))
            argList.append([weightingVars, maritz2018, name, False])
            tempRes = weightHelvestonData(weightingVars, maritz2018, name, save=False)
            helvestonResults[(name, tuple(weightingVars), maritz2018)] = tempRes

        print('Completed weighting Helveston respondents')

        if (save):
            helvestonDFDict = constructFullWeightMappingDF(helvestonResults)
            saveNewWeights(helvestonDFDict, helveston=True)

    if (helvestonMTurk):
        print('Weighting Helveston MTurk repsondents')
        argList = []
        for weightingVars, maritz2018, name in zip(weightingVarLists, maritz2018List, names):
            print('Working on weighting the following demographics: {}'.format(weightingVars))
            argList.append([weightingVars, maritz2018, name, False])
            tempRes = weightHelvestonMTurkData(weightingVars, maritz2018, name, save=False)
            helvestonMTurkResults[(name, tuple(weightingVars), maritz2018)] = tempRes

        print('Completed weighting Helveston respondents')

        if (save):
            helvestonMTurkDFDict = constructFullWeightMappingDF(helvestonMTurkResults)
            saveNewWeights(helvestonMTurkDFDict, helvestonMTurk=True)

    r = {'pooled':pooledResults, 'dynata':dynataResults, 'mturk':mturkResults, 'helveston': helvestonResults, 'helvestonMTurk':helvestonMTurkResults}

    return r


def saveNewWeights(newWeightsDict, pooled=False, mturk=False, dynata=False, helveston=False, helvestonMTurk=False):

    if(pooled):

        if (sys.platform == 'darwin'):
            file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Pooled/PooledWeighted.xlsx'
        else:
            file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Pooled\PooledWeighted.xlsx'
        print('Loading full Pooled data workbook.')
        fullWorkbook = loadPooledWorkbook()

    if(mturk):

        if (sys.platform == 'darwin'):
            file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/MTurk/MTurkDataWeighted.xlsx'
        else:
            file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\MTurk\MTurkDataWeighted.xlsx'

        print('Loading full MTurk data workbook.')
        fullWorkbook = loadMTurkWorkbook()

    if(dynata):

        if (sys.platform == 'darwin'):
            file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Dynata/DynataWeighted.xlsx'
        else:
            file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Dynata\DynataWeighted.xlsx'

        print('Loading full Dynata data workbook.')
        fullWorkbook = loadDynataWorkbook()

    if(helveston):

        if (sys.platform == 'darwin'):
            file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Helveston/helvestonDataWeighted.xlsx'
        else:
            file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Helveston\helvestonDataWeighted.xlsx'

        print('Loading full helveston data workbook.')
        fullWorkbook = loadHelvestonWorkbook()

    if (helvestonMTurk):

        if (sys.platform == 'darwin'):
            file = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Helveston/helvestonDataMTurkWeighted.xlsx'
        else:
            file = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Helveston\helvestonDataMTurkWeighted.xlsx'

        print('Loading full helveston data workbook.')
        fullWorkbook = loadHelvestonMTurkWorkbook()

    newData = {}
    print('Appending weights across workbook')

    for type, idWeightDF in newWeightsDict.items():
        weightSheet = 'weight-{}'.format(type)
        cbcShortSheet = 'cbcShort-{}'.format(type)
        cbcLongSheet = 'cbcLong-{}'.format(type)
        demoSheet = 'demo-{}'.format(type)

        # idWeightDF = weightingOutcomes[type]['mappingDF']

        newData[weightSheet] = deepcopy(fullWorkbook[weightSheet])
        newData[weightSheet] = newData[weightSheet].merge(idWeightDF, on='ID', how='right')

        newData[cbcShortSheet] = deepcopy(fullWorkbook[cbcShortSheet])
        newData[cbcShortSheet] = newData[cbcShortSheet].merge(idWeightDF, on='ID', how='right')

        newData[cbcLongSheet] = deepcopy(fullWorkbook[cbcLongSheet])
        newData[cbcLongSheet] = newData[cbcLongSheet].merge(idWeightDF, on='ID', how='right')

        newData[demoSheet] = deepcopy(fullWorkbook[demoSheet])
        newData[demoSheet] = newData[demoSheet].rename({'id': 'ID'}, axis=1)
        newData[demoSheet] = newData[demoSheet].merge(idWeightDF, on='ID', how='right')

    print('Saving new workbook')
    excelWriter = p.ExcelWriter(file)
    for sheet, data in newData.items():
        data.to_excel(excelWriter, sheet, index=False)

    excelWriter.close()
    print('Workbook saved')

def getCovidOutcomeSet(data):
    r = []

    columns = list(data.columns)
    covidColInd = columns.index('Clean covid')
    for row in data.itertuples(index=False):
        covidVal = row[covidColInd]
        if(not covidVal==covidVal):
            covidVal = 'None of the above'

        r.extend(covidVal.split(';'))

    return set(r)

def getCovidProportions(data, covidVals, weight=None):
    if(weight==None):
        weight = 'Weight'
        data.loc[:, weight] = 1

    total = sum(data.loc[:, weight])

    r = {}

    for covidVal in covidVals:
        r[covidVal] = 0

    columns = list(data.columns)
    covidColInd = columns.index('Clean covid')
    weightInd = columns.index(weight)
    for row in data.itertuples(index=False):
        indCovidVal = row[covidColInd]
        indWeight = row[weightInd]
        if (not indCovidVal == indCovidVal):
            indCovidVal = 'None of the above'

        for covidVal in covidVals:
            if(covidVal in indCovidVal):
                r[covidVal] = r[covidVal]+indWeight

    for covidVal in covidVals:
        r[covidVal] = r[covidVal]/total*100

    return r



def summarizeCovidOutcomes(file=None, pooled=False, typePass = None):
    mturkUnweightedFile = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/MTurk/MTurkDataFull.xlsx'
    mturkWeightedFile = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/MTurk/MTurkDataWeighted.xlsx'
    dynataUnweightedFile = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Dynata/Dynata.xlsx'
    dynataWeightedFile = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Dynata/DynataWeighted.xlsx'


    if (sys.platform == 'darwin'):
        mturkUnweightedFile = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/MTurk/MTurkDataFull.xlsx'
        mturkWeightedFile = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/MTurk/MTurkDataWeighted.xlsx'
        dynataUnweightedFile = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Dynata/Dynata.xlsx'
        dynataWeightedFile = '/Users/connorforsythe/Library/CloudStorage/Box-Box/CMU/EV Preferences Project/Python Code/DataRestructuring/Data/Dynata/DynataWeighted.xlsx'
    else:
        mturkUnweightedFile = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\MTurk\MTurkDataFull.xlsx'
        mturkWeightedFile = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\MTurk\MTurkDataWeighted.xlsx'
        dynataUnweightedFile = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Dynata\Dynata.xlsx'
        dynataWeightedFile = r'C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Dynata\DynataWeighted.xlsx'


    files = [mturkUnweightedFile, mturkWeightedFile, dynataUnweightedFile, dynataWeightedFile]

    if(pooled):
        files = [r"C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Pooled\PooledWeighted.xlsx",r"C:\Users\Connor\Box\CMU\EV Preferences Project\Python Code\DataRestructuring\Data\Pooled\Pooled.xlsx"]
        files.reverse()

    r = {}
    for file in tqdm(files):
        base = os.path.split(file)[1][:os.path.split(file)[1].index('.')]
        if('MTurk' in file):
            types = ['car', 'suv']
        else:
            types = ['car', 'suv', 'truck']

        if(typePass!=None):
            types = typePass

        if('Weight' in file):
            weight = 'Weight2018'
        else:
            weight = None

        for vehType in tqdm(types, disable=True):
            tempData = p.read_excel(file, 'demo-{}'.format(vehType))
            if(vehType==types[0]):
                data = tempData
            else:
                data = data.append(tempData, sort=True)

        covidOutcomes = getCovidOutcomeSet(data)
        covidProps = getCovidProportions(data, covidOutcomes, weight)
        r[base] = covidProps

    r = p.DataFrame.from_dict(r)

    if(pooled):
        renameDict = {'Pooled':'\\makecell{Pooled Sample \\\\ Unweighted}',
                      'PooledWeighted':'\\makecell{Pooled Sample \\\\ Weighted}'}
    else:
        renameDict = {'Dynata': '\\makecell{Dynata Sample \\\\ Unweighted}',
                      'DynataWeighted': '\\makecell{Dynata Sample \\\\ Weighted}',
                      'MTurkData': '\\makecell{MTurk Sample \\\\ Unweighted}',
                      'MTurkDataWeighted': '\\makecell{MTurk Sample \\\\ Weighted}'}

    r = r.rename(renameDict, axis=1)

    if(not pooled):
        r = r.sort_values(list(renameDict.values())[0])
    else:
        r = r.sort_values(list(renameDict.values())[0])

    if(pooled):
        r.to_latex('Tables/covidOutcomesPooled.tex', escape=False, float_format='%9.1f')
    else:
        r.to_latex('Tables/covidOutcomes.tex', escape=False, float_format='%9.1f')

    return r

def loadMaritzData(maritz2018, vehType):
    if(maritz2018):
        r = loadMaritz2018(vehType)
    else:
        r = loadMaritz2010(vehType)

    return r


weightVars = ['Decade', 'Income']
weightVars = [weightVars, weightVars]
weightNames = ['Weight2018', 'Weight2010']
maritz2018List = [True, False]
sumVars = ['Age', 'Single', 'Income', 'CollegeGrad', 'Woman', 'householdSize']

mturk=True
dynata=True
helveston=True
pooled=True
helvestonMTurk = True

#Weight all data to the two baseline demographic sources
t0 = datetime.now()
r = weightMultiple(weightVars, maritz2018List, weightNames, mturk=mturk, dynata=dynata, helveston=helveston, helvestonMTurk=helvestonMTurk, pooled=pooled)
t1 = datetime.now()
diff = t1-t0
timeToWeight = diff.seconds/60
print('Weighting procedure took {} minutes'.format(np.round(timeToWeight,1)))

