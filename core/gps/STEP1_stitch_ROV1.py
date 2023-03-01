"""
STEP1_clean_ROV1.py -- Step 1 in processing continuous GPS data. 

This stage produces a uniformly-sampled time-series of position and uncertainties from 

This conducts the following steps:
A) Convert from Lat/Lon/Elevation into East/North/Up local coordinates in meters
B) Downsample data to a uniform sampling rate of 1 sec to match late-deployment decrease in sampling rate
C) Trim out gaps during instrument reinstallations and wind-gusts & model estimated displacements during instrument reinstallation steps


AUTH: Nathan T. Stevens
EMAIL: ntstevens@wisc.edu
"""
import pandas as pd
import matplotlib.pyplot as plt
import SG_Seven_Decades.util.GPS_utils as gutil
import os

### Data & Metadata Path Definitions ###
# Data Directory Relative Root Path
DROOT = os.path.join('..','..','..','..')
# Input Data File (csv format)
IDATA = os.path.join(DROOT,'data','MINDSatUW','GPS','POST_PROCESSED','continuous','rov1_cat.pos')
# Manually characterized reinstallation and gust data gaps (csv format)
MDATA = os.path.join(DROOT,'data','MINDSatUW','GPS','Meta','continuous','rov1_stitching_gaps.csv')
# Output Data Files (csv format)
OLDATA = os.path.join(DROOT,'processed_data','gps','continuous','S1A_rov1_localized.csv')
ORDATA = os.path.join(DROOT,'processed_data','gps','continuous','S1C_rov1_downsampled.csv')
OSDATA = os.path.join(DROOT,'processed_data','gps','continuous','S1B_rov1_stitched.csv')

### Processing Run parameters ###
# Maximum Quality Level (<= 1 means "fixed")
Qmax = 1
# Column names for mean values
cxyz = ['mE','mN','mZ']
# Column names for 1-StDev values
oxyz = ['sde(m)','sdn(m)','sdu(m)']
# Control parameters for gap correction
gap_kwargs = {'param_wind':pd.Timedelta(1,'h'),\
			  'method':'mean_velocity',\
			  'cxyz':cxyz,'oxyz':oxyz}
# Resampling Period
dt_new = pd.Timedelta(1,unit='sec')
# Output Controls
nplot = 0
verb = 1


### Processing Section ###
# Load Reinstallation Time Gap Metadata File
df_gaps = pd.read_csv(MDATA,parse_dates=['gstart','gstop']).sort_values('gstart')
# Load Raw Data
df_GPS = pd.read_csv(IDATA,parse_dates=True,index_col='GPST')
# Remove Low Quality Data
df_GPS = df_GPS[df_GPS['Q']<=Qmax]

#### STEP 1A: Get Localized Coordinates ####

if verb >= 1:
	print('Data Loaded')

# Convert from LLH to XYZ array
tmp = gutil.global2local(df_GPS['longitude(deg)'].values, df_GPS['latitude(deg)'].values, df_GPS['height(m)'].values,\
                        df_GPS['longitude(deg)'].values[0], df_GPS['latitude(deg)'].values[0],\
                        df_GPS['height(m)'].values[0], crfun = None)

# Format converted XYZ data as a pandas.DataFrame
tmp = pd.DataFrame(tmp,columns=cxyz,index=df_GPS.index)

if verb >= 1:
	print('Converted to UTM')

# Concatenate data to include metric uncertainties, data quality, and number of input samples
df_GPS = pd.concat([tmp,df_GPS.filter(like='sd'),df_GPS[['Q','ns']]],axis=1,ignore_index=False)
# Save Output Localized DATA
df_GPS.to_csv(OLDATA,index=True,header=True)




#### STEP 1B Apply Shift Corrections and Trim Out Data Where Specified ###
if verb >= 1:
	print('Modelling GPS Gaps')
df_GPS, df_MOD = gutil.model_gaps(df_GPS,df_gaps,**gap_kwargs)

### Save Raw-Stitched GPS To File ###
df_GPS.to_csv(OSDATA,index=True,header=True)


#### STEP 1C: Conduct Uniform Resampling ####
# Resample Position Varibles with a mean filter under the assumption of normally distributed values
df_POS = df_GPS[['mE','mN','mZ']].copy().resample(dt_new).mean()
# Resample Cd**0.5, solution qualities, and satellite counts under the assumption of non-normally distributed values
df_COV = df_GPS[['sdn(m)','sde(m)','sdu(m)','sdne(m)','sdeu(m)','sdun(m)','Q','ns']].copy().resample(dt_new).median()
# Reconstitute
df_GPSr = pd.concat([df_POS,df_COV],axis=1,ignore_index=False)
# Save
df_GPSr.to_csv(ORDATA,index=True,header=True)


### Plotting Section ###
# Display displacement data and gaps where shifts occur and those were noisy data were removed
if nplot >= 2:
	T_ = 0
	F_ = 0
	plt.figure()
	plt.plot(df_GPS[cxyz])
	for i_ in range(len(df_gaps)):
		gb = df_gaps['model'].iloc[i_]
		if gb:
			if T_ == 0:
				plt.fill_between(df_gaps.iloc[i_].filter(like='gs').values,[-1.5,-1.5],[1.5,1.5],color='r',alpha=0.25,label='Clip + Shift')
				T_ += 1
			else:
				plt.fill_between(df_gaps.iloc[i_].filter(like='gs').values,[-1.5,-1.5],[1.5,1.5],color='r',alpha=0.25)

		elif not gb:
			if F_ == 0:
				plt.fill_between(df_gaps.iloc[i_].filter(like='gs').values,[-1.5,-1.5],[1.5,1.5],color='b',alpha=0.25,label='Clip Only')
				F_ += 1
			else:
				plt.fill_between(df_gaps.iloc[i_].filter(like='gs').values,[-1.5,-1.5],[1.5,1.5],color='b',alpha=0.25)
	plt.ylim([-2,2])
	plt.legend()