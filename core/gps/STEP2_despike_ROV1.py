"""
STEP2_despike_ROV1.py - conduct iterative despiking of displacement timeseries to remove wind-gust artifacts

This is the second step in processing the ROV2 GPS data

The goal of this script is to detect outliers from wind gusts using a moving-window z-score
data transform to detect impulsive events with durations of a few to tens of seconds. We use
an iterative "despiking" (removing data corresponding to high z-scores) algorithm that was
parameterized as shown below with trial-and-error testing to achieve similar results to 
analyst-based feature removal.

These events are ascribed to strong wind gusts that were observed causing temporarily 
deflections of the GPS antenna followed by a rapid re-settling. 


"""

import pandas as pd
from SG_Seven_Decades.util.DF_utils import get_df_zscore
import os

### Data Path Definitions ###
# Data Directory Relative Root Path
DROOT = os.path.join('..','..','..','..')
# Input Data File (csv format)
IDATA = os.path.join(DROOT,'processed_data','gps','continuous','S1C_rov1_downsampled.csv')
# Output Data File (csv format)
ODATA = os.path.join(DROOT,'processed_data','gps','continuous','S2C_rov1_despiked.csv')


### Processing Control Section ###
## Even Sampling Kwargs
# Resample at base sampling rate to populate missing data points prior to despiking
eswl = 1000		# Even Sampling Window Length
eswu = 'ms' 	# Even Sampling Window Unit
## Iterative Despiking Hyper-Parameter Controls
NITER = 2 		# Number of despiking iterations to 
max_sigma = 3   # Threshold value for z-scores
## Iterative Despiking Window Parameter Controls
gdz_kwargs = {'rwl':20,'rwu':'min','ofmt':'dist'}
## Column Names For Processing
cxyz = ['mE','mN','mZ']
oxyz = ['sde(m)','sdn(m)','sdu(m)']


### Load Data ###
df_GPS = pd.read_csv(IDATA,parse_dates=True,index_col='GPST')
print('Data Loaded...Beginning Filtering')


### DO ITERATIVE DESPIKING ###
for i_ in range(NITER):
	print('Conducting Despiking Iteration {} of {}'.format(i_+1,NITER))
	# Ensure data are evenly sampled
	df_GPS = df_GPS.resample('%d%s'%(eswl,eswu)).mean()
	
	# Get Z-scores
	zsi = get_df_zscore(df_GPS,cxyz,**gdz_kwargs)
	# Remove entries with z-scores greater than the max_sigam threshold
	df_GPS = df_GPS[df_GPS.index.isin(zsi[zsi <= max_sigma].index)]

# Update time-index to correct for application of the resample() method
if eswl != 200:
	df_GPS.index += pd.Timedelta(0.5*eswl,unit=eswu)

### SAVE TO DISK ###
print('Despiking Complete. Saving Results')
df_GPS.to_csv(ODATA,index=True,header=True)
