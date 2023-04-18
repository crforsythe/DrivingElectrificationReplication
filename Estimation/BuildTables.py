from getPythonModels import *
import numpy as np
import pandas as p
from scipy.stats import norm as normDist
from collections import OrderedDict
from helpFile import getTableFromModels, joinDicts, buildFullTable

print('Starting')
# Comparison of different heterogeneity specifications for primary specification
pooledCarLinearModels = loadPooledModels(['linear', 'car', '2018'], addMeanName=True)
pooledSUVLinearModels = loadPooledModels(['linear', 'suv', '2018'], addMeanName=True)
helvestonCarLinearModels = loadHelvestonModels(['linear', 'car', '2010'], addMeanName=True)
helvestonSUVLinearModels = loadHelvestonModels(['linear', 'suv', '2010'], addMeanName=True)

tabPooledCarLinear = getTableFromModels(pooledCarLinearModels)
tabPooledSUVLinear = getTableFromModels(pooledSUVLinearModels)
tabHelvestonCarLinear = getTableFromModels(helvestonCarLinearModels)
tabHelvestonSUVLinear = getTableFromModels(helvestonSUVLinearModels)


tabPooledCarLinear.to_latex('Tables/pooledCarLinear.tex', index=False, escape=False)
tabPooledSUVLinear.to_latex('Tables/pooledSUVLinear.tex', index=False, escape=False)
tabHelvestonCarLinear.to_latex('Tables/helvestonCarLinear.tex', index=False, escape=False)
tabHelvestonSUVLinear.to_latex('Tables/helvestonSUVLinear.tex', index=False, escape=False)

# Comparison of different heterogeneity specifications for alternative specification
pooledCarAsinhModels = loadPooledModels(['asinh', 'car', '2018'], addMeanName=True)
pooledSUVAsinhModels = loadPooledModels(['asinh', 'suv', '2018'], addMeanName=True)
helvestonCarAsinhModels = loadHelvestonModels(['asinh', 'car', '2010'], addMeanName=True)
helvestonSUVAsinhModels = loadHelvestonModels(['asinh', 'car', '2010'], addMeanName=True)

tabPooledCarAsinh = getTableFromModels(pooledCarAsinhModels)
tabPooledSUVAsinh = getTableFromModels(pooledSUVAsinhModels)
tabHelvestonCarAsinh = getTableFromModels(helvestonCarAsinhModels)
tabHelvestonSUVAsinh = getTableFromModels(helvestonSUVAsinhModels)

tabPooledCarAsinh.to_latex('Tables/pooledCarAsinh.tex', index=False, escape=False)
tabPooledSUVAsinh.to_latex('Tables/pooledSUVAsinh.tex', index=False, escape=False)
tabHelvestonCarAsinh.to_latex('Tables/helvestonCarAsinh.tex', index=False, escape=False)
tabHelvestonSUVAsinh.to_latex('Tables/helvestonSUVAsinh.tex', index=False, escape=False)

# Models of Differences in Primary Specification
#Simple logit
pooledSimpleCarModels = loadPooledModels(['linear', 'car', '2018', 'simple'], addMeanName=True)
pooledSimpleSUVModels = loadPooledModels(['linear', 'suv', '2018', 'simple'], addMeanName=True)
helvestonSimpleCarModels = loadHelvestonModels(['linear', 'car', '2010', 'simple'], addMeanName=True)
helvestonSimpleSUVModels = loadHelvestonModels(['linear', 'suv', '2010', 'simple'], addMeanName=True)

simpleCarModels = joinDicts(helvestonSimpleCarModels, pooledSimpleCarModels)
simpleSUVModels = joinDicts(helvestonSimpleSUVModels, pooledSimpleSUVModels)

simpleCarTests = buildFullTable(simpleCarModels, diff=True)
simpleSUVTests = buildFullTable(simpleSUVModels, diff=True)

simpleCarTests.to_latex('Tables/simpleCarTestOverTime.tex', index=False, escape=False)
simpleSUVTests.to_latex('Tables/simpleSUVTestOverTime.tex', index=False, escape=False)

#Mixed logit
pooledMixedCarModels = loadPooledModels(['linear', 'car', '2018', 'mixed'], addMeanName=True)
pooledMixedSUVModels = loadPooledModels(['linear', 'suv', '2018', 'mixed'], addMeanName=True)
helvestonMixedCarModels = loadHelvestonModels(['linear', 'car', '2010', 'mixed'], addMeanName=True)
helvestonMixedSUVModels = loadHelvestonModels(['linear', 'suv', '2010', 'mixed'], addMeanName=True)

mixedCarModels = joinDicts(helvestonMixedCarModels, pooledMixedCarModels)
mixedSUVModels = joinDicts(helvestonMixedSUVModels, pooledMixedSUVModels)

mixedCarTests = buildFullTable(mixedCarModels, diff=True)
mixedSUVTests = buildFullTable(mixedSUVModels, diff=True)

mixedCarTests.to_latex('Tables/mixedCarTestOverTime.tex', index=False, escape=False)
mixedSUVTests.to_latex('Tables/mixedSUVTestOverTime.tex', index=False, escape=False)

#Comparison of range parameterizations
#2021 Study
pooledCarModels = loadPooledModels(['weight2018', 'car'], ['part'])
pooledSUVModels = loadPooledModels(['weight2018', 'suv'], ['part'])

pooledCarTable = getTableFromModels(pooledCarModels)
pooledSUVTable = getTableFromModels(pooledSUVModels)

pooledCarTable.to_latex('Tables/carPooledFunctionalFormComparison2021.tex', index=False, escape=False)
pooledSUVTable.to_latex('Tables/suvPooledFunctionalFormComparison2021.tex', index=False, escape=False)

#2015 Study
helvestonCarModels = loadHelvestonModels(['weight2010', 'car'], ['part'])
helvestonSUVModels = loadHelvestonModels(['weight2010', 'suv'], ['part'])

helvestonCarTable = getTableFromModels(helvestonCarModels)
helvestonSUVTable = getTableFromModels(helvestonSUVModels)

helvestonCarTable.to_latex('Tables/carHelvestonFunctionalFormComparison2021.tex', index=False, escape=False)
helvestonSUVTable.to_latex('Tables/suvHelvestonFunctionalFormComparison2021.tex', index=False, escape=False)

#MTurk Comparison
#Simple logit
mTurkSimpleCarModels = loadPooledMTurkModels(['linear', 'car', '2018', 'simple'], addMeanName=True)
mTurkSimpleSUVModels = loadPooledMTurkModels(['linear', 'suv', '2018', 'simple'], addMeanName=True)
helvestonMTurkSimpleCarModels = loadHelvestonMTurkModels(['linear', 'car', '2010', 'simple'], addMeanName=True)
helvestonMTurkSimpleSUVModels = loadHelvestonMTurkModels(['linear', 'suv', '2010', 'simple'], addMeanName=True)

simpleCarModels = joinDicts(mTurkSimpleCarModels, helvestonMTurkSimpleCarModels)
simpleSUVModels = joinDicts(mTurkSimpleSUVModels, helvestonMTurkSimpleSUVModels)

simpleCarTests = buildFullTable(simpleCarModels, diff=True)
simpleSUVTests = buildFullTable(simpleSUVModels, diff=True)

simpleCarTests.to_latex('Tables/simpleMTurkCarTestOverTime.tex', index=False, escape=False)
simpleSUVTests.to_latex('Tables/simpleMTurkSUVTestOverTime.tex', index=False, escape=False)

#Mixed logit
mTurkMixedCarModels = loadPooledMTurkModels(['linear', 'car', '2018', 'mixed'], addMeanName=True)
mTurkMixedSUVModels = loadPooledMTurkModels(['linear', 'suv', '2018', 'mixed'], addMeanName=True)
helvestonMTurkMixedCarModels = loadHelvestonMTurkModels(['linear', 'car', '2010', 'mixed'], addMeanName=True)
helvestonMTurkMixedSUVModels = loadHelvestonMTurkModels(['linear', 'suv', '2010', 'mixed'], addMeanName=True)

mixedCarModels = joinDicts(mTurkMixedCarModels, helvestonMTurkMixedCarModels)
mixedSUVModels = joinDicts(mTurkMixedSUVModels, helvestonMTurkMixedSUVModels)

mixedCarTests = buildFullTable(mixedCarModels, diff=True)
mixedSUVTests = buildFullTable(mixedSUVModels, diff=True)

mixedCarTests.to_latex('Tables/mixedMTurkCarTestOverTime.tex', index=False, escape=False)
mixedSUVTests.to_latex('Tables/mixedMTurkSUVTestOverTime.tex', index=False, escape=False)


#MTurk-Dynata Comparison
#Simple logit
mTurkSimpleCarModels = loadPooledMTurkModels(['linear', 'car', '2018', 'simple'], addMeanName=True)
mTurkSimpleSUVModels = loadPooledMTurkModels(['linear', 'suv', '2018', 'simple'], addMeanName=True)
dynataSimpleCarModels = loadPooledDynataModels(['linear', 'car', '2018', 'simple'], addMeanName=True)
dynataSimpleSUVModels = loadPooledDynataModels(['linear', 'suv', '2018', 'simple'], addMeanName=True)

simpleCarModels = joinDicts(mTurkSimpleCarModels, dynataSimpleCarModels)
simpleSUVModels = joinDicts(mTurkSimpleSUVModels, dynataSimpleSUVModels)

simpleCarTests = buildFullTable(simpleCarModels, diff=True)
simpleSUVTests = buildFullTable(simpleSUVModels, diff=True)

simpleCarTests.to_latex('Tables/simpleMTurkDynataCarTestOverTime.tex', index=False, escape=False)
simpleSUVTests.to_latex('Tables/simpleMTurkDynataSUVTestOverTime.tex', index=False, escape=False)

#Mixed logit
mTurkMixedCarModels = loadPooledMTurkModels(['linear', 'car', '2018', 'mixed'], addMeanName=True)
mTurkMixedSUVModels = loadPooledMTurkModels(['linear', 'suv', '2018', 'mixed'], addMeanName=True)
dynataMixedCarModels = loadPooledDynataModels(['linear', 'car', '2018', 'mixed'], addMeanName=True)
dynataMixedSUVModels = loadPooledDynataModels(['linear', 'suv', '2018', 'mixed'], addMeanName=True)

mixedCarModels = joinDicts(mTurkMixedCarModels, dynataMixedCarModels)
mixedSUVModels = joinDicts(mTurkMixedSUVModels, dynataMixedSUVModels)

mixedCarTests = buildFullTable(mixedCarModels, diff=True)
mixedSUVTests = buildFullTable(mixedSUVModels, diff=True)

mixedCarTests.to_latex('Tables/mixedMTurkDynataCarTestOverTime.tex', index=False, escape=False)
mixedSUVTests.to_latex('Tables/mixedMTurkDynataSUVTestOverTime.tex', index=False, escape=False)
