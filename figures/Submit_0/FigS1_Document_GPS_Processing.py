import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

### Saving Controls ###
issave = True
FMT = 'png'
DPI = 200
PIN = 0.05
### Data & Metadata Path Definitions ###
# Data Directory Relative Root Path
DROOT = os.path.join('..','..','..')
# Raw Data (csv format)
LOC_DATA = os.path.join(DROOT,'processed_data','gps','rov1','S0_rov1_localized.csv')
# Manually characterized reinstallation and gust data gaps (csv format)
GAP_DATA = os.path.join(DROOT,'data','MINDSatUW','GPS','Meta','rov1','rov1_stitching_gaps.csv')
# Stitched Data (csv format)
STITCH_DATA = os.path.join(DROOT,'processed_data','gps','rov1','S1_rov1_stitched.csv')
# Despiked Data (csv format)
DESP_DATA = os.path.join(DROOT,'processed_data','gps','rov1','S2_rov1_despiked.csv')

#### DATA LOADING SECTION ####
# Conversion factor to local time
IDX_DT = pd.Timedelta(-7,unit='hours')
# Safety catch - prevent re-loading data if already read into memory
# try:
# 	if isloaded:
# 		print('Data already loaded')
# except NameError:
# Load Reinstallation Time Gap Metadata File
df_gaps = pd.read_csv(GAP_DATA,parse_dates=['gstart','gstop']).sort_values('gstart')
print("Reinstallation Gap Times Loaded")
# Load Raw Data
df_0 = pd.read_csv(LOC_DATA,parse_dates=True,index_col='GPST')
# Immediately Remove Low Quality Data
df_0 = df_0[df_0['Q']<=1]
print("Raw Data Loaded and Filtered")
# Load Stitched Data (and downsample to 1 second to match despiked saved data)
df_1 = pd.read_csv(STITCH_DATA,parse_dates=True,index_col='GPST').resample('1s').apply(np.nanmean)
# Do correction of index
df_1.index += pd.Timedelta(0.5,unit='sec')
print("Stitched Data Loaded")
# Load Despiked Data
df_2 = pd.read_csv(DESP_DATA,parse_dates=True,index_col='GPST')
print("Despiked Data Loaded")
	# # Safety catch 
	# isloaded=True

# Adjust DateTimes to local
df_0.index += IDX_DT
df_1.index += IDX_DT
df_2.index += IDX_DT


fig = plt.figure(figsize=[8.4,9.2])
ax0 = fig.add_subplot(311)
ax1 = fig.add_subplot(312,sharex=ax0)
ax2 = fig.add_subplot(313,sharex=ax0)

ax0.plot(df_0['mE'],'k-',label='East')
ax0.plot(df_0['mN'],'r-',label='North')
ax0.set_title('A')
ax0.set_ylabel('Displacement (m)')
ax0.set_ylim([-2,1])
ax0.xaxis.set_visible(False)
ax0.legend()
# plt.legend()

ax1.plot(df_1['mE'],'k-',label='East')
ax1.plot(df_1['mN'],'r-',label='North')
# ax1.set_xlabel('UTC DateTime')
ax1.set_title('B')
ax1.set_ylabel('Displacement (m)')
ax1.set_ylim([-0.1,2.0])
ax1.xaxis.set_visible(False)
ax1.legend()

ax2.plot(df_2['mE'],'k-',label='East')
ax2.plot(df_2['mN'],'r-',label='North')
# ax1.set_xlabel('UTC DateTime')
ax2.set_title('C')
ax2.set_ylabel('Displacement (m)')
ax2.set_xlabel('Local Time (UTC - 7)')
ax2.set_ylim([-0.1,2.0])
ax2.legend()

if issave:
	plt.savefig(os.path.join(DROOT,'results','Supplement','FigS1_Documented_GPS_Processing_%ddpi.%s'%(DPI,FMT.lower())),dpi=DPI,pad_inches=PIN,format=FMT.lower())
