"""
STEP4_estimate_vels_ROV1.py - This is the 4th and final step for processing ROV1 data from Saskatchewan Glacier
This script estimates the long-term 

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import SG_Seven_Decades.util.DF_utils as dutil
import os

### Data Path Definitions ###
# Data Directory Relative Root Path
DROOT = os.path.join('..','..','..','..')
# Input Data File (csv format)
IDATA = os.path.join(DROOT,'processed_data','gps','continuous','S3C_rov1_rotated.csv')
# Temporary file-save locations used by dutil.rolling_linear_WLS()
TLOC = os.path.join(DROOT,'processed_data','gps','continuous','tmp','STEP4C_Velocity')
TDATA = os.path.join(TLOC,'S4_vmod_')
# Output Data File (csv format)
ODATA = os.path.join(DROOT,'processed_data','gps','continuous','S4C_rov1_vels.csv')

try:
	os.mkdir(TLOC)
except FileExistsError:
	print('Temporary save location directory already exists. Previous data may be overwritten')



### PROCESSING PARAMETERS ###
# Time unit conversion (sec/yr)
spyr = 3600*24*365.24
# dutil.rolling_linear_WLS() input parameters
flds = {'x':('mX','vxx'),'y':('mY','vyy'),'z':('mZ','vzz')}
win_len = pd.Timedelta(6,unit='hour')
win_step = pd.Timedelta(3,unit='min')
min_frac=0.75
wpos = 'center'

# Package kwargs
rlWLS_kwargs = {'flds':flds,'wl':win_len,'ws':win_step,'min_frac':min_frac,'wpos':wpos,'tmp_save_pre':TDATA}


# Load Data from Stage 3
df_GPS = pd.read_csv(IDATA,parse_dates=True,index_col=0)
print('Data Loaded')



### Calculate Long Term Velocity (V_LT) Vectors
cxyz = ['mX','mY','mZ']	# Columns to estimate velocities from
IDX0 = df_GPS['vxx'].notna()
df_mods = pd.DataFrame()
dt_yr = ((df_GPS[IDX0].index - df_GPS[IDX0].index.min()).total_seconds()/spyr)
print('Estimating Long-Term Velocities')
wd_ = {'mX':'vxx','mY':'vyy','mZ':'vzz'}
for c_ in cxyz:
	data = df_GPS[IDX0][c_].values
	# 1/var[c_] for weights, per np.polyfit() 
	wgts = df_GPS[IDX0][wd_[c_]].values
	imod,icov = dutil.curve_fit(dutil.lin_fun,dt_yr,data,sigma=wgts)
	# imod,icov = np.polyfit(dt_yr,data,1,w=wgts,cov='unscaled')
	idf = pd.DataFrame({'m(m/yr)':imod[0],'b(m)':imod[1],\
						'smm(m/yr)':icov[0,0]**0.5,'sbb(m)':icov[1,1]**0.5,'vmb(m2/yr2)':icov[0,1]},index=[c_])
	df_mods = pd.concat([df_mods,idf],axis=0)

### REPORT V_LT \pm 2-\sigma ###
print('Long Term Velocity Vectors')
print(df_mods)
print('Horizontal Velocity Magnitude')
print('%.3f +\- %.3f m/yr'%((df_mods.loc[['mX','mY'],'m(m/yr)']**2).sum()**0.5,\
							   (df_mods.loc[['mX','mY'],'smm(m/yr)']**2).sum()**0.5))

### Calculate moving window (short-term) velocity vectors ###
df_VEL = dutil.rolling_linear_WLS(df_GPS,**rlWLS_kwargs)


# SAVE TO FILE
df_VEL.to_csv(ODATA,header=True,index=True)