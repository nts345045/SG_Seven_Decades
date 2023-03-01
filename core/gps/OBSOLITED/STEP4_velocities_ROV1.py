"""
STEP4_estimate_vels_ROV1.py - This is the 4th and final step for processing ROV1 data from Saskatchewan Glacier
This script estimates the long-term 

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import SG_Seven_Decades.util.GPS_utils as gutil
import os

### Data Path Definitions ###
# Data Directory Relative Root Path
DROOT = os.path.join('..','..','..','..')
# Input Data File (csv format)
IDATA = os.path.join(DROOT,'processed_data','gps','rov1','S3_rov1_rotated.csv')
# Output Data File (csv format)
ODATA = os.path.join(DROOT,'processed_data','gps','rov1','S4_rov1_vels.csv')


### PROCESSING PARAMETERS ###
# Time unit conversion (sec/yr)
secpyr = 3600*24*365.24
# Downsampling Period Prior to Velocity Estimation (speeds up calculation)
dsT = '20s'
# Velocity Estimation Parameters
cxyz = ['mX','mY','mZ']	# Columns to estimate velocities from
vxyz = ['vxx','vyy','vzz'] # Columns containing location variances
RW = 180				  	# [int] Scalar for velocity model window length
RU = 'min' 			  	# [str] Unit for velocity model window length
MV_kwargs = {'Clip':0.1,\
			 'wskw':{'min_pct':0.75},\
			 'PolyOpt':True,\
			 'PolyMaxOrder':4,\
			 'verb':True}


### LOAD DATA ###
df_GPS = pd.read_csv(IDATA,parse_dates=True,index_col=0)
print('Data Loaded')

### Calculate Long Term Velocity (V_LT) Vectors with WLS
IDX0 = df_GPS['vxx'].notna()
df_mods = pd.DataFrame()
dt_yr = ((df_GPS[IDX0].index - df_GPS[IDX0].index.min()).total_seconds()/secpyr)
print('Estimating Long-Term Velocities')
wd_ = {'mX':'vxx','mY':'vyy','mZ':'vzz'}
for c_ in cxyz:
	data = df_GPS[IDX0][c_].values
	# 1/var[c_] for weights, per np.polyfit() 
	wgts = df_GPS[IDX0][wd_[c_]].values**-1
	imod,icov = np.polyfit(dt_yr,data,1,w=wgts,cov='unscaled')
	idf = pd.DataFrame({'m(m/yr)':imod[0],'b(m)':imod[1],\
						'smm(m/yr)':icov[0,0]**0.5,'sbb(m)':icov[1,1]**0.5,'vmb(m2/yr2)':icov[0,1]},index=[c_])
	df_mods = pd.concat([df_mods,idf],axis=0)

### REPORT V_LT \pm 2-\sigma ###
print('Long Term Velocity Vectors')
print(df_mods)
print('Horizontal Velocity Magnitude')
print('%.3f +\- %.3f m/yr'%((df_mods.loc[['mX','mY'],'m(m/yr)']**2).sum()**0.5,\
							   (df_mods.loc[['mX','mY'],'smm(m/yr)']**2).sum()**0.5))

### DOWNSAMPLE DATA ###
df_GPS = df_GPS.resample(dsT).mean()
print('Data Downsampled!')




## DO VELOCITY MODELING
df_VELS = gutil.model_velocities_v2(df_GPS[cxyz],df_GPS[cxyz],RW,RU,**MV_kwargs)

## DO 

# SAVE TO FILE
df_VELS.to_csv(ODATA,header=True,index=True)