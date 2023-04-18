This set of files provides the replication code for all major results in
"Driving Electric Vehicle Adoption: The Role of Technology and Consumer Preferences"
by Connor R. Forsythe, Kenneth T. Gillingham, Jeremy J. Michalek, and Kate S. Whitefoot

*****Replication Information*****
Due to the limitations of GitHub storage, not all replication data can be stored in a single repository. Therefore, only the primary analysis data are provided. In order to obtain the full results that have already been estimated, please visit https://cmu.box.com/s/2zme67dl7agnstxbkhr1wcaeag0lvga1 for the entire codebase and results. 

*****Script Information*****

This folder contains the necessary data and code to recreate all primary results contained in "Driving Electric Vehicle Adoption: The Role of Technology and Consumer Preferences" that don't require access to proprietary data. Note, running the full code may take several days. Currently, mixed logit estimation will require the most compute resources. Luckily the xlogit package leverage GPU processing. Still, tuning of specific parameters (described below) will likely be required in order to best utilize GPU. 


- Directories
	-- ASC: Data for market simulation 
	-- Tables: Output tables are located here
	-- Data: Several necessary data as well as output are stored here
	-- HeadToHeadSims: Results of head-to-head simulations are stored here
	-- Models: Results of logit estimation are stored here
	-- Plots: Output plots are located here
- Estimation Scripts
	-- genWeightingDistribution.py: This is the only script that will not execute as it involves the use of propietary data (see note below). However, it is the script used to estimate consumer weights based on Maritz survey data for both the newly-collected data and Helveston, et al. (2015) data.
	-- CrossValidation-Car.py: Estimates 5-fold cross-validation in-sample models for car-buyers.
	-- CrossValidation-SUV.py: Estimates 5-fold cross-validation in-sample models for suv-buyers.
	-- EstimatDynataModels.py: Estimates logit models when considering only the newly-collected, Dynata data.
	-- EstimatMTurkModels.py: Estimates logit models when considering only the newly-collected, Mechanical Turk data.
	-- EstimatPooledModels.py: Estimates logit models when considering all newly-collected data.
	-- EstimatPooledUnweightedModels.py: Estimates logit models when considering all newly-collected data without weighting.
	-- EstimatPooledCovidModels.py: Estimates logit models when considering newly-collected data that were not economically affected by the COVID-19 pandemic.
	-- EstimateHelvestonModels.py: Estimates logit models using all Helveston, et al. (2015) data.
	-- EstimateHelvestonMTurkModels.py: Estimates logit models using only Helveston, et al. (2015) Mechanical Turk data.
	-- simulateHeadToHeadShares.py: Esimates head-to-head shares acrsoss
- Output Scripts
	-- buildBreakdown.py: Outputs the "breakdown" plots, such as Fig. S6.
	-- buildCovidComparisonPlots.py: Outputs plots that 
	-- buildPlots.py: Build several timeline and waterfall plots available in the SI.
	-- buildTables.py: Build tables outputting primary parameter estimates as well as tests comparing attributes over time.
	-- plotMarketSimOutcomes: Plots simulation results
	-- plotHeadToHeadSimulations: Plots outcomes of choice simulations calculated 
	-- CrossValidationPrediction.py: Outputs the summary statistics table for cross-validation exercises.
- Additional Analysis Scripts
- Utility Scripts
	-- helpFile.py: Contains various utility functions used throughout the above files.
	-- tableHelpFile.py: Contain various utility functions used throughout the table-building files.
	-- GetPHEVTripDist.py: Provides functions to return PHEV electric utilization rates.
	-- getPythonModels.py: Allows for the reading of estimated models.
	-- Inflator.py: Contains an object that can easily adjust prices for inflation.


*****Results Produced with Propietary Data*****
Although we strive for transparency and replicability, certain aspects of our
analysis cannot be shared with the public. Namely, our weighting procedure is dependent on
propietary data that cannot be distributed. Therefore, these results cannot be
replicated. However, we do provide the code used to generate these weights in the Python script
"genWeightingDistribution.py".

*****Software Requirements*****
All provided scripts, except for those that require propietary data, only require a Python environment to run.

***Python Package Requirements***
-numpy;
-scipy;
-matplotlib;
-pandas;
-seaborn;
-xlxswriter;
-xlogit;