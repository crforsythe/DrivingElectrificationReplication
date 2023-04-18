import pandas as p
import numpy as np
import sys
class Inflator(object):
    def __init__(self):
        if(sys.platform=='darwin'):
            self.allUnadjustedFile = r'Data/CPI/SeriesReport-20210114225605_0bed88.xlsx'
            self.allUnadjustedFile = r'Data/CPI/SeriesReport-20220519161028_41520e.xlsx'
        else:
            self.allUnadjustedFile = r"Data\CPI\SeriesReport-20210114225605_0bed88.xlsx"
            self.allUnadjustedFile = r"Data\CPI\SeriesReport-20220519161028_41520e.xlsx"


        self.allUnadjustedData = self.cleanData(self.allUnadjustedFile)


    def inflateAll(self, price, fromYear, toYear):
        if(fromYear in self.allUnadjustedData.keys() and toYear in self.allUnadjustedData.keys()):
            yearInd = self.allUnadjustedData[fromYear]
            toInd = self.allUnadjustedData[toYear]

            newPrice = price*(toInd/yearInd)

            return newPrice
        else:
            print('One of the years are not in range of keys.')

    def inflateGas(self, price, year, to):
        if (year in self.gasUnadjustedData.keys() and to in self.gasUnadjustedData.keys()):
            yearInd = self.gasUnadjustedData[year]
            toInd = self.gasUnadjustedData[to]

            newPrice = price*(toInd / yearInd)

            return newPrice
        else:
            print('One of the years are not in range of keys.')

    def inflateMotorFuel(self, price, year, to):
        if (year in self.motorFuelUnadjustedData.keys() and to in self.motorFuelUnadjustedData.keys()):
            yearInd = self.motorFuelUnadjustedData[year]
            toInd = self.motorFuelUnadjustedData[to]

            newPrice = price*(toInd / yearInd)

            return newPrice
        else:
            print('One of the years are not in range of keys.')

    def inflateVehicle(self, price, year, to):
        if (year in self.vehicleUnadjustedData.keys() and to in self.vehicleUnadjustedData.keys()):
            yearInd = self.vehicleUnadjustedData[year]
            toInd = self.vehicleUnadjustedData[to]

            newPrice = price * (toInd / yearInd)

            return newPrice
        else:
            print('One of the years are not in range of keys.')

    def inflateMaintenance(self, price, year, to):
        if (year in self.maintenanceUnadjustedData.keys() and to in self.maintenanceUnadjustedData.keys()):
            yearInd = self.maintenanceUnadjustedData[year]
            toInd = self.maintenanceUnadjustedData[to]

            newPrice = price * (toInd / yearInd)

            return newPrice
        else:
            print('One of the years are not in range of keys.')

    def inflateGDP(self, price, year, to):
        if (year in self.gdpInflatorData.keys() and to in self.gdpInflatorData.keys()):
            yearInd = self.gdpInflatorData[year]
            toInd = self.gdpInflatorData[to]

            newPrice = price * (toInd / yearInd)

            return newPrice
        else:
            print('One of the years are not in range of keys.')

    def inflateMaintenanceMonth(self, price, yearFrom, monthFrom, toYear):

        monthDict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

        fromKey = (yearFrom, monthFrom)

        if (fromKey in self.maintenanceUnadjustedDataByMonth.keys() and toYear in self.maintenanceUnadjustedData.keys()):
            fromInd = self.maintenanceUnadjustedDataByMonth[fromKey]
            toInd = self.maintenanceUnadjustedData[toYear]

            newPrice = price * (toInd / fromInd)

            return newPrice
        else:
            print('One of the years are not in range of keys.')

    def cleanData(self, file):
        preData = p.read_excel(file)

        years = list(preData.iloc[11:, 0])
        indices = list(preData.iloc[11:, 13])

        r = dict(zip(years, indices))

        return r

    def cleanDataBEA(self, file, sheet='T10109-A'):
        preData = p.read_excel(file, sheet, skiprows=7)
        yearStartCol = list(preData.columns).index('1929')

        indices = np.array(list(preData.iloc[0,yearStartCol:])).astype(float)
        years = np.array(list(preData.columns)[yearStartCol:]).astype(int)

        r = dict(zip(years, indices))

        return r

    def cleanDataMonth(self, file):
        preData = p.read_excel(file)

        rows, cols = preData.shape

        yearCol = 0
        monthRow = 10

        years = list(preData.iloc[11:, 0])
        months = list(preData.iloc[10, 1:13])

        monthDict = dict(zip(list(range(1,13)), months))
        monthDictRev = dict(zip(monthDict.values(), monthDict.keys()))
        indices = list(preData.iloc[11:, 13])

        rowRange = range(11, rows)
        colRange = range(1, 13)

        r = {}

        for row in rowRange:
            tempYear = preData.iloc[row, yearCol]
            for col in colRange:
                tempMonth = monthDictRev[preData.iloc[monthRow, col]]
                tempIndex = preData.iloc[row, col]
                tempKey = (tempYear, tempMonth) #Keys are pairs of year-month in tuples

                r[tempKey] = tempIndex


        return r